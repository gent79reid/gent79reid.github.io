---
title: "inet_bind ( ) function"
date: 2011-12-21 14:10:00 +0800
categories: [Linux]
tags: [Linux,C Programming]
---

the call flow of bind( ) function can boil down to a terse flow like this:  sys_socketcall ( ) —> sys_bind ( ) —> inet_bind ( );  I took a snippet of inet_bind ( ) function below:

```c
int inet_bind(struct socket *sock, struct sockaddr *uaddr, int addr_len)
{
	struct sockaddr_in *addr = (struct sockaddr_in *)uaddr;
	struct sock *sk = sock->sk;
	struct inet_sock *inet = inet_sk(sk); 
	unsigned short snum;
	int chk_addr_ret;
	int err;
............................
static inline struct inet_sock *inet_sk(const struct sock *sk)
{
	return (struct inet_sock *)sk;
}
struct inet_sock {
	/* sk and pinet6 has to be the first two members of inet_sock */
	struct sock sk;
#if defined(CONFIG_IPV6) || defined(CONFIG_IPV6_MODULE)
	struct ipv6_pinfo	*pinet6;
#endif
	/* Socket demultiplex comparisons on incoming packets. */
	__be32			daddr;
	__be32			rcv_saddr;
	__be16			dport;
	__u16			num;
	__be32			saddr;
	__s16			uc_ttl;
	__u16			cmsg_flags;
	struct ip_options	*opt;
	__be16			sport;
	__u16			id;
	__u8			tos;
	__u8			mc_ttl;
	__u8			pmtudisc;
	__u8			recverr:1,
				is_icsk:1,
				freebind:1,
				hdrincl:1,
				mc_loop:1;
	int			mc_index;
	__be32			mc_addr;
	struct ip_mc_socklist	*mc_list;
	struct {
		unsigned int		flags;
		unsigned int		fragsize;
		struct ip_options	*opt;
		struct dst_entry	*dst;
		int			length; /* Total length of all frames */
		__be32			addr;
		struct flowi		fl;
	} cork;
};
```
sock structure has been converted to inet_sock, and the first member in inet_sock is struct sock. the instance of *sk can fit into inet_sock well. Because as a normal programming principle, abstracting common part and putting into a separate structure, could make the code neat and more easily to use it in somewhere else.

struct inet_sock *inet = inet_sk(sk);” 

But I didn’t see any code to allocate memory for those members after sk. If inet invoke some unallocated member, like “inet -> rcv_saddr”, any affect ?


