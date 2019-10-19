---

layout: post
title: "Running Cisco Trex"
description: "This post will cover some basic usages of Cisco Trex, show you how to simply launch Trex to validate your DUT in home lab."
categories: [Tools]
tags: [Cisco,Trafffic Generator]
comments: true
redirect_from:
  - /2019/10/18/
---

**Side Note** This post does not aim to articulate the power of Cisco Trex, and not cover its advance features. I will keep updating this post series as I am going deeper in my project, which will heavily rely on Trex to do all kinds of interesting design. Trex doesn't have very fancy GUI client like Spirent, but its GUI client is getting improved a lot. 

# Install Cisco Trex 

The quickest way to test Trex is to running its docker version, which might not be the latest version. 

```bash
➜# docker search trex
NAME                               DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
trexcisco/trex                     Official's Cisco TRex traffic generator for …   7
trexcisco/trex-dev                 Official TRex development environment           1
```

In this test, I am not catering for testing its performance, as all components in my lab are hosting on my PC with only 32G, 8 x Core CPU. Please navigate the official installation guide [Cisco Trex Installation Guide](https://trex-tgn.cisco.com/trex/doc/trex_manual.html/)

# Some Trex Terminologies 

If you don't have solid appreciation of these terms, you might run into some troubles while playing Trex, I suggest to take some time to understand them before you go. 

**Stateless**

Mainly for testing L2 & L3 functions, relevant mostly for a switch or router. Remember that in stateless mode, there is no concepts of flow/client/server. You can define a stream with one packet template, sending from one port to another one on the same Trex instance. 

**Stateful**

Trex has stateful and advanced stateful mode, you can think of stateful testing like generating some flows with the real application behavior, it is useful for testing application-aware DUT, like NAT, IPS, Load-balancer, etc. The easiest way to use stateful testing is to replay the recorded pcap onto the wire, with **Trex field engine** you can alter src/dst IP address of the packets on the fly. 

**Profile**

Profile is the piece of Python code, which uses Scapy module to build packets. The profile can be invoked directly from Trex interactive mode with **start** command. 

The below is a profile sample:

```py
[root@3f9e8965416c v2.41]# cat stl/flow_stats_latency.py
from trex_stl_lib.api import *

class STLS1(object):
    """
    Create flow stat latency stream of UDP packet.
    Can specify using tunables the packet length (fsize) and packet group id (pg_id)
    Since we can't have two latency streams with same pg_id, in order to be able to start this profile
    on more than one port, we add port_id to the pg_id
    Notice that for perfomance reasons, latency streams are not affected by -m flag, so
    you can only change the pps value by editing the code.
    """

    def __init__ (self):
        self.fsize = 64
        self.pg_id = 0

    def _create_stream (self):
        size = self.fsize - 4; # HW will add 4 bytes ethernet FCS
        base_pkt = Ether() / IP(src = "16.0.0.1", dst = "48.0.0.1") / UDP(dport = 12, sport = 1025)
        pad = max(0, size - len(base_pkt)) * 'x'
        pkt = STLPktBuilder(pkt = base_pkt/pad)

        return [STLStream(packet = pkt,
                          mode = STLTXCont(pps=1000),
                          flow_stats = STLFlowLatencyStats(pg_id = self.pg_id))
               ]

    def get_streams (self, fsize = 64, pg_id = 7, **kwargs):
        self.fsize = fsize
        self.pg_id = pg_id + kwargs['port_id']
        return self._create_stream()


# dynamic load - used for trex console or simulator
def register():
    return STLS1()
```

Start Trex server:

```
[bash]>sudo ./t-rex-64 -i
```

Attach local python client to the server:

```bash
[bash]>trex-console                                                    
```

Run the above stream:

```bash
trex>start -f stl/udp_1pkt_simple.py -m 10mbps -a 
```
To verify network convergence, use **stateless mode** plus rx stats feature. 

# Using Trex to generate simple traffic

Here is the lab which I am building to verify Cisco Viptela SD-WAN solution, the lab is running on the top of EVE-NG, I have to admit that eve-ng is very user-friendly, easy learning-curve and excellent UI to draw your arbitrary topology. After paid for the professional license, you can enjoy docker and more fancy topology drawing, most importantly PRO version has integrated the link quality simulation, saving your time to use [linux-netem](https://wiki.linuxfoundation.org/networking/netem) or [wanem](http://wanem.sourceforge.net/). 

![image-20191019155002861](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019155002861.png)

Please avoid the part of SD-WAN components, we will only focus on Cisco Trex on the right-hand side, I will write other posts to walk through the steps of setting up Cisco Viptela SD-WAN on all virtualized environment. 

In this lab, I installed Cisco Trex docker and imported to EVE-NG environment. The Trex docker has bee assigned 4 ports, one of them is used to communicate with Trex GUI client which is running on my Mac. Normally we should allocate even number of ports to Trex, but here I will use two of them at a time. 

To start with one simple stateless testing, Trex will generate a stream of UDP traffic with source ip address ( 172.16.150.2/24 at Site 150) and destination ip address ( 172.16.200.2/24 at Site 200), this stream is **unidirectional** from Trex eth2 to eth3, I want to verify the statistics of packet loss while Branch 1 vedge's Internet circuit went down. This is a simple test we always do in network convergence timer testing.  

I will use [Trex GUI client](https://github.com/cisco-system-traffic-generator/trex-stateless-gui) for better presenting the testing result. 

### Start Trex server with default configuration file.

   ```bash
   [root@Cisco-Trex v2.41]# cat /etc/trex_cfg.yaml 
   - port_limit      : 2
     version         : 2
   #List of interfaces. Change to suit your setup. Use ./dpdk_setup_ports.py -s to see available options
     interfaces    : ["eth2","eth3"] 
     port_info       :  # Port IPs. Change to suit your needs. In case of loopback, you can leave as is.
             - ip         : 172.16.150.2
               default_gw : 172.16.150.1
             - ip         : 172.16.200.2
               default_gw : 172.16.200.1
   
   [root@Cisco-Trex v2.41]# ./t-rex-64 -i
   Killing Scapy server... Scapy server is killed
   Starting Scapy server.... Scapy server is started
   The ports are bound/configured.
   ```

### Open Trex GUI client and connect to the server:

   ![image-20191019155022256](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019155022256.png)

### Acquire Port 0 which is mapping to eth2, port 1 to eth3.

   ![image-20191019154829306](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019154829306.png)

### Convert Python code to YAML file **(Profile)**, and add the profile associating with Port 0 

```bash
[root@Cisco-Trex v2.41]# ./stl-sim -f stl/flow_stats_latency.py --yaml > stl/flow_stats_latency.yaml 
[root@Cisco-Trex v2.41]# cat stl/flow_stats_latency.yaml 
- action_count: 0
  enabled: true
  flags: 0
  flow_stats:
    enabled: true
    rule_type: latency
    stream_id: 7
  isg: 0.0
  mode:
    rate:
      type: pps
      value: 1000
    type: continuous
  packet:
    binary: !!python/unicode 'AAAAAQAAAAAAAgAACABFAAAuAAEAAEARxJisEJYCrBDIAgQBAAwAGglLeHh4eHh4eHh4eHh4eHh4eHh4'
    meta: ''
  self_start: true
  start_paused: false
  vm:
    instructions: []
```

### ***Traffic Profiles*** -> ***load Profile***  load up the profile 

![image-20191019175826840](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019175826840.png)

Change Rate parameter to "pps/1.0k", so we can circulate the time of network convergency. (one packet loss = the 1ms loss).

**Important:**  <u>Check RX Stats and Latency enabled, PG ID could be assigned to integer and used to track multiple streams  </u>

### Use Packet Editor to alter packet content, remove default TCP layer and add UDP layer, also update src/dst IP address.

![image-20191019164800639](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019164800639.png)

### Now, it's time to fire up Trex

   ![image-20191019180212740](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019180212740.png)

Navigate to "Latency" tab, check the counters of "dropped".

![image-20191019180405409](/assets/images/2019-10-18-Run_Cisco_Trex/image-20191019180405409.png)









