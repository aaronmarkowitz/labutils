#!/bin/bash
# filepath: ~/minimal_working_ioc.sh

# Kill all existing softIOC processes and zombies
sudo pkill -9 -f softIoc 2>/dev/null || true
sleep 1

# Clean environment
rm -rf /tmp/auxioc
mkdir -p /tmp/auxioc

# Create a much simpler startup script
cat > /tmp/auxioc/st.cmd << 'EOF'
#!/bin/sh
softIocPVA -D /usr/lib/epics/dbd -d /usr/lib/epics/dbd/softIoc.dbd
epicsEnvSet("EPICS_CA_SERVER_PORT","5064")
epicsEnvSet("EPICS_CA_ADDR_LIST", "127.0.0.1")
epicsEnvSet("EPICS_CA_AUTO_ADDR_LIST", "NO")

# Must load all record types before loading any databases
dbLoadDatabase
softIoc_registerRecordDeviceDriver(pdbbase)

# Now load the actual databases
dbLoadRecords("/home/controls/labutils/leybold_turbolab.db", "P=Y1:,R=AUX-,DESC=,EGU=")
dbLoadRecords("/home/controls/labutils/mokuIOC.db", "P=Y1:,R=AUX-,DESC=,EGU=")
dbLoadRecords("/home/controls/labutils/teem_laser.db", "P=Y1:,R=AUX-,DESC=,EGU=")

iocInit
dbl
EOF

chmod +x /tmp/auxioc/st.cmd

# Run the IOC directly
exec /usr/bin/softIoc -S /tmp/auxioc/st.cmd
