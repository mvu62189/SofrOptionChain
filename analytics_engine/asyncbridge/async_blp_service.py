# async_blp_service.py
"""
Asynchronous Bloomberg Service using an asyncio bridge.

This class runs the blocking blpapi.Session in a separate thread
and pushes all events into an asyncio.Queue for safe
consumption by an async main thread.
"""

import blpapi
import asyncio
import threading
from typing import Union, Any

# --- We will need these later. Let's define them now. ---
REF_DATA_SERVICE = "//blp/refdata"
MKT_DATA_SERVICE = "//blp/mktdata"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
SUBSCRIPTION_DATA = blpapi.Name("SubscriptionData")
SUBSCRIPTION_FAILURE = blpapi.Name("SubscriptionFailure")
RESPONSE = blpapi.Name("Response")
PARTIAL_RESPONSE = blpapi.Name("PartialResponse")
RESPONSE_ERROR = blpapi.Name("responseError")
SECURITY_DATA = blpapi.Name("securityData")
SECURITY_ERROR = blpapi.Name("securityError")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
ERROR_INFO = blpapi.Name("errorInfo")


class AsyncBloombergService:
    def __init__(self):
        """
        Initialize the service and the bridge queue.
        """
        print("Initializing AsyncBloombergService...")
        self.session = None
        self.refDataService = None
        
        # The bridge: An asyncio.Queue to pass events
        # from the poller thread to the main async loop.
        self.bridge_queue = asyncio.Queue()
        
        # A flag to signal the poller thread to stop
        self._poller_running = threading.Event()

    async def start(self) -> bool:
        """
        Starts the session synchronously, opens services.
        This is run once at the beginning.
        """
        print("Starting session...")
        options = blpapi.SessionOptions()
        options.setServerHost("localhost")
        options.setServerPort(8194)
        
        # We MUST create the session in synchronous mode (eventHandler=None)
        self.session = blpapi.Session(options)
        
        if not self.session.start():
            print("Failed to start session.")
            return False
        
        print("Session started.")

        if not self.session.openService(REF_DATA_SERVICE):
            print(f"Failed to open service: {REF_DATA_SERVICE}")
            return False
        self.refDataService = self.session.getService(REF_DATA_SERVICE)
        print(f"Service {REF_DATA_SERVICE} opened.")

        if not self.session.openService(MKT_DATA_SERVICE):
            print(f"Failed to open service: {MKT_DATA_SERVICE}")
            return False
        print(f"Service {MKT_DATA_SERVICE} opened.")
        return True

    def start_poller_thread(self):
        """
        Launches the synchronous poller function in a new,
        daemonized thread.
        """
        print("Starting BLPAPI poller thread...")
        self._poller_running.set() # Set the flag to True
        
        # Get the running asyncio loop
        loop = asyncio.get_running_loop()
        
        # Create and start the poller thread
        poller_thread = threading.Thread(
            target=self._run_poller_loop,
            args=(loop,),
            daemon=True  # Daemon stops thread when main program exits
        )
        poller_thread.start()
        print("Poller thread started.")

    def _run_poller_loop(self, loop: asyncio.AbstractEventLoop):
        """
        This is the synchronous function that runs in the new thread.
        Its ONLY job is to poll for events and put them on the queue.
        """
        print("Poller thread: Loop started. Waiting for events...")
        while self._poller_running.is_set():
            try:
                # This is a BLOCKING call.
                # It's okay because it's in its own thread.
                event = self.session.nextEvent(timeout=1000)

                if event.eventType() == blpapi.Event.TIMEOUT:
                    continue # Just loop again
                
                # We have a real event. Put it on the async queue.
                # We must use call_soon_threadsafe to safely
                # pass the event from this thread to the main async loop.
                loop.call_soon_threadsafe(self.bridge_queue.put_nowait, event)

            except Exception as e:
                print(f"Poller thread error: {e}")
                
        print("Poller thread: Loop finished.")

    def stop(self):
        """
        Stops the session and signals the poller thread to exit.
        """
        print("Stopping service...")
        if self.session:
            self._poller_running.clear() # Signal the poller to stop
            self.session.stop()
            print("Session stopped.")

    # --- We will add other functions here later ---
    # async def subscribe(self, ...):
    # async def get_initial_snapshot(self, ...):
    # async def _process_events(self, ...):
    # async def _get_on_demand_snapshot(self, ...):

    # --- NEW ASYNC SNAPSHOT FUNCTIONS (START) ---

    async def get_snapshot_data(self, tickers: list[str], fields: list[str], overrides: list[tuple] = None) -> dict[str, Any]:
        """
        Public async wrapper for making a snapshot request.
        
        This runs the blocking C++ call in a separate thread
        to keep the main asyncio loop free.
        """
        print(f"Async: Requesting snapshot for {len(tickers)} tickers...")
        try:
            # asyncio.to_thread runs the given blocking function
            # in a separate thread and awaits its result.
            snapshot_data = await asyncio.to_thread(
                self._get_snapshot_data_sync,
                tickers,
                fields,
                overrides or []
            )
            print(f"Async: Received snapshot data.")
            return snapshot_data
        except Exception as e:
            print(f"Async: Snapshot request failed: {e}")
            return {}

    def _get_snapshot_data_sync(self, tickers: list[str], fields: list[str], overrides: list[tuple]) -> dict[str, Any]:
        """
        The private, SYNCHRONOUS function that performs the actual
        blocking call. This runs in a worker thread.
        """
        
        # 1. Create a private, one-time-use EventQueue.
        # This is our "private mailbox" to get the reply.
        queue = blpapi.EventQueue()
        
        # 2. Create the request
        request = self.refDataService.createRequest("ReferenceDataRequest")
        for ticker in tickers:
            request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)

        # Add any overrides (e.g., for OPT_CHAIN)
        if overrides:
            override_element = request.getElement("overrides")
            for key, val in overrides:
                ovr = override_element.appendElement()
                ovr.setElement("fieldId", key)
                ovr.setElement("value", val)

        # 3. Send the request and tell BLPAPI to send the
        # reply to our private 'queue', NOT the main session.
        self.session.sendRequest(request, eventQueue=queue)

        # 4. Block and wait for the reply on our private queue.
        # This is safe because we are in a worker thread.
        results = {}
        while True:
            # This nextEvent() is on the QUEUE, not the SESSION.
            # It will NOT steal events from our main poller.
            event = queue.nextEvent()
            
            # (A robust error-handling parser should go here)
            # For now, we just parse the data:
            if event.eventType() in (RESPONSE, PARTIAL_RESPONSE):
                for msg in event:
                    if not msg.hasElement(SECURITY_DATA):
                        continue # Skip error messages

                    sec_data_array = msg.getElement(SECURITY_DATA)
                    
                    for i in range(sec_data_array.numValues()):
                        sec_data = sec_data_array.getValue(i)
                        ticker = sec_data.getElementAsString("security")
                        field_data = sec_data.getElement("fieldData")
                        
                        ticker_results = {}
                        for field in fields:
                            if field_data.hasElement(field):
                                ticker_results[field] = field_data.getElement(field).getValue()
                            else:
                                ticker_results[field] = None
                        results[ticker] = ticker_results
            
            if event.eventType() == RESPONSE:
                break # Final response, we're done
                
        return results

    # --- NEW ASYNC SNAPSHOT FUNCTIONS (END) ---