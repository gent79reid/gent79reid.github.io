---
title: Building Splunk Attack Range V3.0 on WSL2
author: gent79
date: 2024-05-01
categories: [Splunk]
tags: [Lab,Cybersecurity]
share: true
---
# Why on WSL2 ? 
At the time of writing this, Splunk Attack Range V3.0 is currently compatible only with local setups on MacOS and Linux. Naturally, leveraging AWS or Azure would offer greater ease and scalability, albeit with budget considerations. However, given my well-equipped laptop boasting 48GB of RAM, why not maximize its potential for my threat hunting endeavors? Alright then, let's dive right in!
# Steps of bring it alive
## Bring up Ubuntu 22.04 WSL2 
Just follow up the link to setup WSL2 
[Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install)
Highly recommend you to move your WSL2 image to non-C driver
```shell
wsl --export <DistributionName> <PathToExport>
wsl --unregister <DistributionName>
wsl --import <NewDistributionName> <NewInstallLocation> <PathToExport>
wsl --set-default <NewDistributionName>
```

## Prepare Splunk Attack Range Installation Environment

Install Virtualbox lastest version on Windows Host, there is no need to install Vagrant on Windows Host. 

https://attack-range.readthedocs.io/en/latest/Attack_Range_Local.html#linux
While using Poetry to build the dependency, it would be difficult for some people who has slow Internet access, so you might need to open up the pyproject.toml file and manually install them one by one. 

Next, as the ansible playbooks at current release are still using "include" keyword that is already deprecated, so use below command to replace it.
```bash
find . -name "main.yml" -exec sed -i 's/include:/include_tasks:/g' {} +
```

Then, making sure to enable Windows Access permission for Vagrant, so Vagrant can use VirtualBox on Windows Host as its provider, and install Vagrant WSL2 plugin
```bash
export VAGRANT_WSL_ENABLE_WINDOWS_ACCESS="1"
export PATH="$PATH:/mnt/d/Program Files/Oracle/VirtualBox"
export VAGRANT_WSL_WINDOWS_ACCESS_USER_HOME_PATH="/mnt/d/Github/attack_range/"
vagrant plugin install virtualbox_WSL2
```

Now, You are safe to run 
```
 python attack_range.py configure
 python attack_range.py build
```
Be patient, it's going to take a while, probably 30-40 min depending on how fast you can download Vagrant base images. 

