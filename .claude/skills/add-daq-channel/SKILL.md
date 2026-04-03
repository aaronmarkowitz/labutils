---
name: add-daq-channel
description: Add a new EPICS softIOC channel to the Y1:AUX system and stream it to the CDS DAQ for recording via NDS2/ndscope. Use when adding new EPICS channels that should be recorded to frames.
---

# Adding a new Y1:AUX EPICS channel to the DAQ

Follow these steps to add a new channel to the softIOC and have it recorded to DAQ frames.

## Step 1 — Define the EPICS record in a .db file

Edit or create a `.db` file in `/home/controls/labutils/epics/`. Add a record:

```
record(ao, "$(P)$(R)MY_CHANNEL") {
    field(DESC, "Description")
    field(PREC, "3")
    field(EGU, "units")
    field(PINI, "YES")
    field(VAL, "0")
}
```

Common record types: `ao`/`ai` (float), `bo`/`bi` (binary 0/1), `mbbi` (multi-state enum), `calcout` (computed output with CA link).

**Important**: Do NOT use `SCAN="1 second"` on `bi`/`bo` records written by external services — the periodic scan re-reads INP and resets the value. Use `SCAN="Passive"` (default) instead.

## Step 2 — Load the .db in the softIOC

Edit `/home/controls/labutils/epics/minimal_working_ioc.sh`, add:

```bash
dbLoadRecords("/home/controls/labutils/epics/my_file.db", "P=Y1:,R=AUX-,DESC=,EGU=")
```

Then restart the softIOC on **worker1**:

```bash
sudo systemctl restart auxioc
```

Verify: `caget Y1:AUX-MY_CHANNEL`

## Step 3 — Add channel to edc.ini on cymac1

SSH to cymac1 and append to `/etc/advligorts/edc.ini`:

```
[Y1:AUX-MY_CHANNEL]
datatype=5
units=myunit
```

Datatypes: `5` = float32 (for `ao`/`ai`/temperature), `4` = int32 (for integer/enum channels).

**Note**: Channels written by external code via caput use `datatype=5` for floats, `datatype=4` for integers.
Binary (`bi`/`bo`) channels: use `datatype=4`.

## Step 4 — Restart ALL four DAQ services on cymac1

**Critical**: Restarting only rts-edc is NOT sufficient. All four services must restart together:

```bash
ssh cymac1 "sudo systemctl restart rts-edc.service rts-local_dc.service rts-daqd.service rts-nds.service"
```

Wait ~10 seconds for them to settle.

## Step 5 — Verify

From worker1, check connectivity:
```bash
# Channel is live
caget Y1:AUX-MY_CHANNEL

# cymac1 can see it
ssh cymac1 "caget Y1:AUX-MY_CHANNEL"

# Check NDS2 (use GPS time, not Unix time!)
python3 -c "
import nds2, time
conn = nds2.connection('192.168.1.11', 8088)
GPS_OFFSET = 315964818
gps_now = int(time.time()) - GPS_OFFSET
buf = conn.fetch(gps_now-30, gps_now-3, ['Y1:AUX-MY_CHANNEL'])
arr = buf[0].data
print(f'samples={len(arr)} mean={arr.mean():.3f}')
"
```

Use `/var/lib/cds-conda/base/envs/cds-testing/bin/python3` if nds2 not in path.

## Notes

- EDC status page: http://192.168.1.11:9000/
- edc.ini: dcuid=52, datarate=16 Hz
- NDS2 server: 192.168.1.11:8088
- **NDS2 queries must use GPS time** (GPS = Unix - 315964818), not Unix timestamps
- For calcout records that write to CDS channels via CA: softIOC's CA_ADDR_LIST must include `192.168.1.255` (subnet broadcast) — already configured in minimal_working_ioc.sh
- Documentation: `/home/controls/Downloads/Private & Shared/Stream EPICS data to DAQ *.md`
