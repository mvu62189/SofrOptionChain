import blpapi
from datetime import datetime

# Setup session
session = blpapi.Session()
if not session.start():
    raise RuntimeError("Failed to start session")
if not session.openService("//blp/refdata"):
    raise RuntimeError("Failed to open //blp/refdata")

refdata = session.getService("//blp/refdata")

# Create ReferenceDataRequest for OPT_CHAIN
req = refdata.createRequest("ReferenceDataRequest")
req.append("securities", "SFRU5 Comdty")
req.append("fields", "OPT_CHAIN")

# Optional: add CHAIN_DATE override
ovr = req.getElement("overrides").appendElement()
ovr.setElement("fieldId", "CHAIN_DATE")
ovr.setElement("value", "20240601")  # YYYYMMDD

session.sendRequest(req)

tickers = []

while True:
    ev = session.nextEvent()
    for msg in ev:
        if msg.hasElement("securityData"):
            sec_data_array = msg.getElement("securityData")
            for i in range(sec_data_array.numValues()):
                sec_data = sec_data_array.getValueAsElement(i)
                if sec_data.hasElement("fieldData"):
                    field_data = sec_data.getElement("fieldData")
                    if field_data.hasElement("OPT_CHAIN"):
                        chain_array = field_data.getElement("OPT_CHAIN")
                        for j in range(chain_array.numValues()):
                            opt_elem = chain_array.getValueAsElement(j)
                            if opt_elem.hasElement("Security Description"):
                                tickers.append(opt_elem.getElementAsString("Security Description"))
    if ev.eventType() == blpapi.Event.RESPONSE:
        break

print(f"✅ Retrieved {len(tickers)} option tickers.")
print(tickers[:10])
