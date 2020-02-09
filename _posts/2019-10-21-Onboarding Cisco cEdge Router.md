---
layout: post
title: "Onboarding Cisco cEdge Router"
description: "This lab is to demonstrate the steps of bringing up Cisco CSR1kV without PnP service in lab environment. In real world, it is strongly recommended to leverage Cisco PnP service using auto bootstrap, either by copy bootstrap file to Physical cEdge compatible routers or use cloud-init for Virtual instance. 
"
categories: [SD-WAN]
tags: [Cisco, Viptela, Installation]
comments: true
---

cEdge will automatically initiate PnP server, so I need to disable it right after it boot up completely.

```bash
BR-2-cEdge#pnpa service discovery stop
```

I strongly recommend you to load up following initial configuration in a bulk input, I ran into some weird issues when I typed each configuration sections, especially Tunnel interface. 

```bash

config-transaction
system
 system-ip             192.168.1.5
 domain-id             1
 site-id               200
 admin-tech-on-failure
 organization-name     gent79.me
 vbond vbond.gent79.me
!
ip host vbond.gent79.me 99.0.99.11

interface GigabitEthernet1
 no shutdown
 ip address 99.0.99.25 255.255.255.0
exit
interface GigabitEthernet2
 no shutdown
 ip address 10.0.10.25 255.255.255.0

interface Tunnel1
 no shutdown
 ip unnumbered GigabitEthernet1
 tunnel source GigabitEthernet1
 tunnel mode sdwan
exit
interface Tunnel2
 no shutdown
 ip unnumbered GigabitEthernet2
 tunnel source GigabitEthernet2
 tunnel mode sdwan
exit

sdwan
 interface GigabitEthernet1
  tunnel-interface
   encapsulation ipsec
   color public-internet
   no allow-service bgp
   allow-service dhcp
   allow-service dns
   allow-service icmp
   no allow-service sshd
   no allow-service netconf
   no allow-service ntp
   no allow-service ospf
   no allow-service stun
   no allow-service snmp
  exit
 exit
 interface GigabitEthernet2
  tunnel-interface
   encapsulation ipsec
   color mpls restrict
   max-control-connections 0
   no allow-service bgp
   allow-service dhcp
   allow-service dns
   allow-service icmp
   no allow-service sshd
   no allow-service netconf
   no allow-service ntp
   no allow-service ospf
   no allow-service stun
   no allow-service snmp
  exit
          
```

Copy Root CA certificate to cEdge bootflash and install it:

```bash
BR-1-cEdge#copy scp://root@192.168.31.79//root/sdwan-lab/Gent79_Root_CA.pem bootflash:/root_ca.crt
BR-1-cEdge#request platform software sdwan root-cert-chain install bootflash:/root_ca.crt
Uploading root-ca-cert-chain via VPN 0
Copying ... /bootflash//root_ca.crt via VPN 0
Updating the root certificate chain..
Successfully installed the root certificate chain
```

Assuming that you have uploaded the WAN edge list from Cisco PnP portal to vManage. Pick up one available from the list, the Token is used as temporary code while activating cedge in next step. 

**After uploading WAN Edge list to vManage, don't forget to click the button to push them to vSmart controller. **

![D678DB91-F5F5-4D46-A6A1-6C1A93B851FA](/assets/images/2019-10-21-Onboarding Cisco cEdge Router/D678DB91-F5F5-4D46-A6A1-6C1A93B851FA.png)

```bash
BR-1-cEdge#request platform software sdwan vedge_cloud activate chassis-number CSR-81AFE172-61C8-8CEC-0978-xxxxx token xxxxxxxxxx

BR1-cEdge-01#show sdwan certificate serial 
Chassis number: CSR-81AFE172-61C8-8CEC-0978-A313E45EA070 serial number: xxxxx
```

Next, check that cedge router is in "In Sync" status, and it's time to let vManage take over the control from CLI mode. Before that the relevant templates should be in place. 

![image-20191022204050180](/assets/images/2019-10-21-Onboarding Cisco cEdge Router/image-20191022204050180.png)

![image-20191022204112427](/assets/images/2019-10-21-Onboarding Cisco cEdge Router/image-20191022204112427.png)

Done!

![image-20191022204434426](/assets/images/2019-10-21-Onboarding Cisco cEdge Router/image-20191022204434426.png)

Next post, I will cover the steps to configuring and operating SD-AVC in Viptela solution.

