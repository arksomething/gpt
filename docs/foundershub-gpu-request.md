# Founders Hub request — GPU Startup classification

Submit at `foundershub.startups.microsoft.com` → Support / Contact us.
Azure support **cannot** resolve this; only the Startups program can re-tag the
subscription. Ticket 2607290040012012 was already refused on exactly this basis.

---

**Subject:** Request GPU Startup classification for sponsorship subscription 4a785577-c93b-4bc7-81c8-e4d6939e19ff

**Body:**

Hello,

I'm requesting that my Microsoft for Startups sponsorship subscription be
classified as a **GPU Startup** subscription so that GPU VM families become
available for quota requests.

**Subscription details**
- Subscription ID: `4a785577-c93b-4bc7-81c8-e4d6939e19ff`
- Subscription name: Azure subscription 1
- Account: ark296296@gmail.com
- Offer: Microsoft for Startups sponsorship ($10,000 credits, active)
- Region of interest: East US / East US 2

**What I have already tried**

Four self-service GPU quota requests, all auto-rejected:

| Date | Family | Requested |
|---|---|---|
| 2026-07-28 | NCADSA10v4 | 72 vCPU |
| 2026-07-30 | NCADSA100v4 | 24 vCPU |
| 2026-07-30 | NCADSH100v5 | 40 vCPU |
| 2026-07-31 | NCADSA100v4 | 24 vCPU |

Azure support case **2607290040012012** was then opened and refused on
2026-07-30, with the response: *"Due to high demand for these graphics-enabled
VM types, availability is limited for customers using free/benefit
subscriptions."* The engineer suggested a separate pay-as-you-go subscription.

My understanding is that sponsorship subscriptions are not eligible for GPU
quota until they are tagged as GPU Startup, which is a program-side change
rather than an Azure capacity decision. That is what I am asking for here.

Note the A10 request (NCADSA10v4, a widely available family) was also refused,
which suggests the blocker is subscription eligibility rather than regional
capacity.

**Workload justification**

I am training small language models (25M–1B parameters) as a staged research
project, and I have real, reproducible usage rather than a speculative plan:

- 10 completed training runs (25M parameters, 250M tokens each) on a controlled
  data-mixture experiment, with three baseline seeds establishing measurement
  noise and five bounded perturbations read against it
- A reproducible pipeline: fingerprinted corpora, per-source validation,
  supervised fine-tuning, GRPO reinforcement learning, HF export with verified
  logit and tokenizer parity, and a benchmark harness
- All work to date has run on AWS, where I am currently throttled to 8 GPU
  vCPUs and frequently cannot obtain capacity at all
- Next stage requires 350M- and 1B-parameter runs, which need multi-GPU nodes

The $10,000 in sponsorship credits is currently unusable for this work: the
subscription holds a single storage account and has never been able to launch
a GPU.

**Requests**

1. Tag subscription `4a785577-c93b-4bc7-81c8-e4d6939e19ff` as GPU Startup.
2. Advise which GPU family and region to request once tagged (NCadsA100v4 or
   NCadsH100v5 in East US 2 would suit; A10 or T4 would also be workable).
3. Consider eligibility for the **Startup GPU Cluster** (ND H100 v5 / NDm
   A100 v4), which would fit the 1B-scale runs.
4. Given that credits have been burning down while GPU access has been
   unavailable, please also advise on a **credit expiration extension**.

Thank you,
Evan Liu
ark296296@gmail.com
