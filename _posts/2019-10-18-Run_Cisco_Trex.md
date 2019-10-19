---

layout: post
title: "Run Cisco Trex"
description: "The Deep Dive of Cisco Trex"
categories: [Tools]
tags: [Cisco,Trafffic Generator]
redirect_from:
  - /2019/10/18/
---

[Install Cisco Trex](#Install-Cisco-Trex)

[Some Trex Terminologies](#Some-Trex-Terminologies)

[Using Trex to generate simple traffic](#Using-Trex-to-generate-simple-traffic)



**NOT COMPLETE**

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

 



