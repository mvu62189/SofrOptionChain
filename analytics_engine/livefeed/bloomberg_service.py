# bloomberg_service.py
"""
Manages all communication with the Bloomberg API (blpapi).

This class handles:
- Session startup and teardown.
- Opening services (refdata and mktdata).
- Synchronous reference data requests (for snapshots).
- Asynchronous market data subscriptions (for volume).
- The main event loop.
"""

import blpapi
from blpapi import Event
import config
from data_models import InitialSnapshotModel, LiveSnapshotModel, TradeOccursModel
from typing import List, Dict, Any, Union
from datetime import datetime

# Define Bloomberg service names
REF_DATA_SERVICE = "//blp/refdata"
MKT_DATA_SERVICE = "//blp/mktdata"

# Define Bloomberg event types
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
SUBSCRIPTION_DATA = blpapi.Name("SubscriptionData")
SUBSCRIPTION_FAILURE = blpapi.Name("SubscriptionFailure")
RESPONSE = blpapi.Name("Response")
PARTIAL_RESPONSE = blpapi.Name("PartialResponse")
# Define Message-level error names
RESPONSE_ERROR = blpapi.Name("responseError")
SECURITY_DATA = blpapi.Name("securityData")
SECURITY_ERROR = blpapi.Name("securityError")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
ERROR_INFO = blpapi.Name("errorInfo")  # <--- ADD THIS LINE

class BloombergService:
    """
    A wrapper class to manage a blpapi.Session and simplify data requests.
    """

    def __init__(self, host='localhost', port=8194):
        """
        Initializes the session options.
        """
        print("Initializing BloombergService...")
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(host)
        session_options.setServerPort(port)
        
        self.session = blpapi.Session(session_options)
        self.refdata_service = None
        self.mktdata_service = None

    def start(self) -> bool:
        """
        Starts the session and opens the required services.
        
        Returns:
            True if session and services started successfully, False otherwise.
        """
        print("Starting session...")
        if not self.session.start():
            print("Failed to start session.")
            return False

        # Wait for session to start
        try:
            self._wait_for_event(SESSION_STARTED)
            print("Session started successfully.")
        except InterruptedError as e:
            print(f"Session startup failed: {e}")
            return False

        # Open the reference data service
        if not self.session.openService(REF_DATA_SERVICE):
            print(f"Failed to open service: {REF_DATA_SERVICE}")
            return False
        
        try:
            self.refdata_service = self._wait_for_service_open(REF_DATA_SERVICE)
            print(f"Service {REF_DATA_SERVICE} opened successfully.")
        except InterruptedError as e:
            print(f"Service open failed: {e}")
            return False

        # Open the market data service
        if not self.session.openService(MKT_DATA_SERVICE):
            print(f"Failed to open service: {MKT_DATA_SERVICE}")
            return False
        
        try:
            self.mktdata_service = self._wait_for_service_open(MKT_DATA_SERVICE)
            print(f"Service {MKT_DATA_SERVICE} opened successfully.")
        except InterruptedError as e:
            print(f"Service open failed: {e}")
            return False
            
        return True

    def stop(self):
        """
        Stops the session.
        """
        print("Stopping session...")
        if self.session:
            self.session.stop()
        print("Session stopped.")

    def _wait_for_event(self, event_name: blpapi.Name, timeout_ms: int = 5000):
        """
        Waits for a specific session event (e.g., SessionStarted).
        """
        start_time = datetime.now()
        while True:
            event = self.session.nextEvent(timeout_ms)
            
            # Check for timeout
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed > timeout_ms:
                raise InterruptedError(f"Timeout waiting for {event_name}")

            if event.eventType() == blpapi.Event.SESSION_STATUS:
                for msg in event:
                    if msg.messageType() == event_name:
                        return
                    if msg.messageType() == SESSION_STARTUP_FAILURE:
                        raise InterruptedError(f"Session startup failure: {msg}")

    def _wait_for_service_open(self, service_name: str, timeout_ms: int = 5000) -> blpapi.Service:
        """
        Waits for a specific service to be opened.
        """
        start_time = datetime.now()
        while True:
            event = self.session.nextEvent(timeout_ms)

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed > timeout_ms:
                raise InterruptedError(f"Timeout waiting for {service_name}")

            if event.eventType() == blpapi.Event.SERVICE_STATUS:
                for msg in event:
                    if msg.messageType() == SERVICE_OPENED and \
                       msg.getElementAsString("serviceName") == service_name:
                        return self.session.getService(service_name)
                    if msg.messageType() == SERVICE_OPEN_FAILURE:
                        raise InterruptedError(f"Service open failure: {msg}")

    def get_option_chain(self, underlying_ticker: str) -> List[str]:
        """
        Fetches the entire option chain for a given underlying ticker.
        This version is robust:
        1. Adds the required OVERRIDE to get all options.
        2. Handles all API error messages gracefully to prevent crashes.
        """
        if not underlying_ticker.endswith(" Comdty"):
             underlying_ticker += " Comdty"
             
        print(f"Fetching option chain for {underlying_ticker}...")
        if not self.refdata_service:
            raise RuntimeError("Reference Data Service not open.")
            
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        request.append("securities", underlying_ticker)
        request.append("fields", "OPT_CHAIN")

        # --- THIS IS THE PROCEDURAL FIX ---
        # Add an override to specify *which* options to return (e.g., ALL).
        # This is what high-level libraries like xbbg do automatically.
        overrides = request.getElement("overrides")
        override = overrides.appendElement()
        override.setElement("fieldId", "OPTION_CHAIN_TYPE_OVERRIDE")
        override.setElement("value", "ALL") # Get all options
        # --- END OF FIX ---

        cid = self.session.sendRequest(request)
        chain = []
        
        try:
            while True:
                event = self.session.nextEvent()
                is_final_response = event.eventType() == blpapi.Event.RESPONSE
                
                if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                    
                    for msg in event: 
                        if msg.correlationIds()[0] != cid:
                            continue # Not our message

                        # --- THIS IS THE ROBUSTNESS FIX ---
                        
                        # 1. Check for a Response-level error (bad request)
                        # This catches the 'securityData not found' crash.
                        if not msg.hasElement("securityData"):
                            if msg.hasElement("responseError"):
                                err = msg.getElement("responseError")
                                print(f"[ERROR] ResponseError for {underlying_ticker}: {err.getElementAsString('message')}")
                            else:
                                print(f"[ERROR] Unknown message, no 'securityData': {msg}")
                            break # Stop processing this failed request

                        # 2. If we are here, msg HAS securityData.
                        security_data_array = msg.getElement("securityData")
                        
                        for i in range(security_data_array.numValues()):
                            security_data = security_data_array.getValue(i)
                            
                            # 3. Check for a Security-level error (bad ticker)
                            if security_data.hasElement("securityError"):
                                sec_error = security_data.getElement("securityError")
                                print(f"[ERROR] SecurityError for {underlying_ticker}: {sec_error.getElementAsString('message')}")
                                continue # Skip this one bad security
                                
                            field_data = security_data.getElement("fieldData")
                            
                            # 4. Check for a Field-level error
                            if field_data.hasElement("fieldExceptions"):
                                fld_err_array = field_data.getElement("fieldExceptions")
                                for f_err in fld_err_array.values():
                                     print(f"[ERROR] FieldError for {underlying_ticker}: {f_err.getElement(ERROR_INFO).getElementAsString('message')}")
                                continue # Skip this security

                            # 5. If all checks pass, get the data
                            if field_data.hasElement("OPT_CHAIN"):
                                options_array = field_data.getElement("OPT_CHAIN")
                                for j in range(options_array.numValues()):
                                    option_element = options_array.getValue(j)
                                    ticker = option_element.getElementAsString("Security Description")
                                    cleaned_ticker = ' '.join(ticker.split())
                                    chain.append(cleaned_ticker)
                            else:
                                # This can happen if the ticker is valid but has no options
                                print(f"[WARN] No 'OPT_CHAIN' field returned for {underlying_ticker}, despite no error.")
                
                if is_final_response:
                    break # All messages for this request have been processed
                
        except Exception as e:
            # This catches code bugs, not API errors
            print(f"CRITICAL code error processing option chain: {e}")
            return [] 
            
        print(f"Found {len(chain)} options in chain.")
        return chain

    def get_reference_data(self, tickers: List[str], fields: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Performs a synchronous snapshot request for a list of tickers and fields.
        Used for both the EOD snapshot and the on-demand snapshot.

        Returns:
            A dictionary: {ticker: {field: value, ...}, ...}
        """
        if not self.refdata_service:
            raise RuntimeError("Reference Data Service not open.")
            
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        
        # Append tickers
        for ticker in tickers:
            request.append("securities", ticker)
            
        # Append fields
        for field in fields:
            request.append("fields", field)

        # Send the request
        cid = self.session.sendRequest(request)
        
        results = {}
        try:
            while True:
                event = self.session.nextEvent()
                if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                    for msg in event:
                        if msg.correlationIds()[0] == cid:
                            security_data_array = msg.getElement("securityData")
                            
                            for i in range(security_data_array.numValues()):
                                security_data = security_data_array.getValue(i)
                                ticker = security_data.getElementAsString("security")
                                field_data = security_data.getElement("fieldData")
                                
                                ticker_results = {}
                                for field in fields:
                                    if field_data.hasElement(field):
                                        ticker_results[field] = field_data.getElement(field).getValue()
                                    else:
                                        ticker_results[field] = None # Field not found
                                
                                results[ticker] = ticker_results

                    if event.eventType() == blpapi.Event.RESPONSE:
                        break # Final response
        except Exception as e:
            print(f"Error processing reference data response: {e}")
            
        return results

    def subscribe(self, tickers: List[str], fields: List[str]):
        """
        Subscribes to market data for a list of tickers.
        """
        if not self.mktdata_service:
            raise RuntimeError("Market Data Service not open.")
            
        subscriptions = blpapi.SubscriptionList()
        for i, ticker in enumerate(tickers):
            # Create a unique CorrelationId for each subscription
            corr_id = blpapi.CorrelationId(ticker) 
            subscriptions.add(
                topic=f"{MKT_DATA_SERVICE}/ticker/{ticker}",
                fields=",".join(fields),
                correlationId=corr_id
            )
            print(f"Adding subscription for {ticker}...")

        self.session.subscribe(subscriptions)
        print(f"Sent subscription request for {len(tickers)} tickers.")

    def next_event(self, timeout_ms: int = 100) -> Union[Event, None]:
        """
        Pulls the next event from the session queue.
        This is the heart of the event loop.
        """
        try:
            # nextEvent(timeout) will return None on timeout
            event = self.session.nextEvent(timeout_ms)
            return event
        except Exception as e:
            print(f"Error in next_event: {e}")
            return None