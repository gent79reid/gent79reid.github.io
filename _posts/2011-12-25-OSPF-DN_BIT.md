---
title: OSPF DN-BIT
date: 2011-12-25 14:10:00 +0800
categories: [Cisco,RS]
tags: [Cisco,IOS,IOS-XR]
---



To avoid routing loop in OSPF running between PE and CE, PE sets Down-bit in every LSA sending to CE. CE ignores all LSA with that bit, even through CE has already installed that LSAs into OSPF database.  Any routes sending from PE to  CE may be looped back by any other CE at elsewhere, especially the CE router is acting as PE in other SP network, and the interface is being put in a VRF. So if there’s a demand to install the routes received from PE on a CE router (IOS-XR), just configure the below command in corresponding VRF instance.

```bash
router ospf 1
 router-id 192.168.1.2
 area 0
  interface Loopback0
  !
  interface GigabitEthernet0/0/0/2
   network broadcast
  !
  interface GigabitEthernet0/0/0/3
  !
  interface GigabitEthernet0/0/0/4
  !
 !
 vrf RED_VRF
  router-id 172.16.0.2
  domain-tag 1
  disable-dn-bit-check
  address-family ipv4 unicast
  area 0
   interface Loopback1
   !
   interface GigabitEthernet0/0/0/1
   !
 ```