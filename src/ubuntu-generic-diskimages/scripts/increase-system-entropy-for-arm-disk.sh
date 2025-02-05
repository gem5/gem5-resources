
#!/bin/bash

# Copyright (c) 2025 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Increase system entropy for ARM disk images to reduce boot time in gem5.
# Some systemd services wait for sufficient entropy before proceeding, causing delays.
# By increasing entropy early, these services start without unnecessary waiting.
sudo apt-get install -y haveged

# Update the DAEMON_ARGS environment variable in /etc/default/haveged
echo "Updating DAEMON_ARGS in /etc/default/haveged..."
if grep -q '^#DAEMON_ARGS=""' /etc/default/haveged; then
    # If DAEMON_ARGS is commented out, uncomment and set it
    sudo sed -i 's/#DAEMON_ARGS=""/DAEMON_ARGS="-w 1024"/' /etc/default/haveged
elif grep -q '^DAEMON_ARGS=' /etc/default/haveged; then
    # If DAEMON_ARGS exists, update its value
    sudo sed -i 's/^DAEMON_ARGS=.*/DAEMON_ARGS="-w 1024"/' /etc/default/haveged
else
    # If DAEMON_ARGS is missing, add it
    echo 'DAEMON_ARGS="-w 1024"' | sudo tee -a /etc/default/haveged > /dev/null
fi

# Start the haveged service
echo "Starting the haveged service..."
sudo systemctl start haveged

# Enable the haveged service to start at boot
echo "Enabling the haveged service to start at boot..."
sudo systemctl enable haveged

# Check the status of haveged
echo "Checking the status of the haveged service..."
sudo systemctl status haveged

echo "Haveged installation and setup complete!"
