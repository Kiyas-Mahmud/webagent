# আমাদের failure-aware web agent কীভাবে বানানো হয়েছে — ধাপে ধাপে ব্যাখ্যা

২০২৬-০৯-২৮। এই লেখায় সহজ ভাষায় বলা হয়েছে agent-টা কী, প্রতিটা অংশ কীভাবে কাজ
করে, বানাতে গিয়ে কী সমস্যা হয়েছিল আর কীভাবে ঠিক করা হয়েছে, আর আমাদের system
agent-কে ভালো করে কিনা তা কীভাবে পরীক্ষা করা হবে। সঠিক সংখ্যা আর ফাইলের বিস্তারিত
জানতে লিংক করা technical document-গুলো দেখুন।
(ইংরেজি সংস্করণ: [AGENT_BUILD_EXPLAINED.md](AGENT_BUILD_EXPLAINED.md))

---

## ধাপ ১ — মূল ধারণা: শক্তিশালী agent কাজ করবে, আমাদের model বিচার করবে

আমাদের trained model-এর চারটা pillar আছে (P1–P4)। এর action অংশ (P3: কোথায়
click, type বা scroll করতে হবে) দুর্বল, কারণ এর দেওয়া click-এর জায়গা প্রায়ই ঠিক
element-এ পড়ে না। কিন্তু failure আর recovery অংশ (P1) শক্তিশালী: আমাদের নিজের
validation data-তে এটা সফল আর ব্যর্থ step ভালোভাবে আলাদা করতে পারে (MCC ≈ ০.৬৫)।

তাই agent-কে দুই ভূমিকায় ভাগ করা হয়েছে:

- **Action করবে একটা open-source agent।** আমরা ব্যবহার করছি **Browser Use**, যা
  চালায় untrained **Qwen2.5-VL-7B** base model। এটা page-এর element-কে নম্বর দিয়ে
  চেনে আর সেই নম্বর ধরে click করে, তাই pixel coordinate লাগে না।
- **বিচার আর মনে রাখার কাজ করবে আমাদের trained model।** প্রতিটা action-এর পর এটা
  ঠিক করে step সফল হলো কিনা, কী ধরনের failure হলো, আর কোন recovery strategy
  ভালো (P1)। এটা আগের অভিজ্ঞতা জমা রাখে আর দরকারে খুঁজে বের করে (P4)।

আমাদের model কখনো action-এর নিয়ন্ত্রণ নেয় না। এটা শুধু পরামর্শ দেয়, পরের action
Browser Use নিজেই ঠিক করে।

---

## ধাপ ২ — Agent-এর অংশগুলো

```
                ┌──────────────────────────────────────┐
  আসল website   │  Chromium browser, 1280×720           │
  (Wikipedia,   │  আমাদের dataset-এর site থেকে নেওয়া    │
   arXiv, …)    └───────────▲──────────────┬───────────┘
                            │ action        │ screenshot + page-এর element
                ┌───────────┴──────────────▼───────────┐
  ① কাজ করে     │  Browser Use + Qwen2.5-VL-7B (base)   │
                │  click · type · scroll · select ·     │
                │  navigate · press key                 │
                └───────────┬──────────────▲───────────┘
                            │ আগের ও পরের   │ পরামর্শ
                            │ screenshot    │ (recovery + memory)
                ┌───────────▼──────────────┴───────────┐
  ② বিচার করে   │  আমাদের trained Qwen2.5 model — P1     │
                │  step-এর ফল · failure-এর ধরন ·        │
                │  recovery strategy · recovery-র ফল    │
                └───────────┬──────────────▲───────────┘
                            │ অভিজ্ঞতা      │ মিল থাকা পুরনো অভিজ্ঞতা
                ┌───────────▼──────────────┴───────────┐
  ③ মনে রাখে    │  Experience memory — P4                │
                │  memory head ঠিক করে কী রাখা হবে      │
                └──────────────────────────────────────┘
```

| Pillar | কী করে | Agent-এর কোথায় |
|---|---|---|
| P1 Failure detection ও recovery | প্রতিটা step বিচার করে; recovery strategy বাছে; recovery কাজ করল কিনা দেখে | আমাদের trained head (`assessment.py`) |
| P2 Multimodal বোঝা | screenshot আর text একসাথে বোঝে | একই Qwen model-এর ভেতরে |
| P3 Action | click, type, scroll… | **Browser Use** (আমাদের দুর্বল P3 ব্যবহার হয় না) |
| P4 Memory | failure→recovery অভিজ্ঞতা জমা রাখে ও খুঁজে দেয় | `memory/experience.py` |

---

## ধাপ ৩ — একটা step-এ কী হয়

1. **Browser Use page দেখে** (screenshot আর interactive element-এর তালিকা) একটা
   action বাছে, যেমন *search box-এ "Alan Turing" লেখো*।
2. **Browser action-টা করে**, তারপর নতুন screenshot তোলা হয়।
3. **P1 আগের আর পরের screenshot তুলনা করে** উত্তর দেয়: সফল নাকি ব্যর্থ? কী ধরনের
   failure? কোন recovery strategy (RETRY, REPLAN, BACKTRACK, ALTERNATIVE_TARGET)?
4. **P1 ব্যর্থ বললে:**
   - **P4 memory-তে খোঁজে**, একই page-এ আগে কী ভুল হয়েছিল আর ঠিক করতে কী
     চেষ্টা করা হয়েছিল।
   - Diagnosis আর পাওয়া অভিজ্ঞতা Browser Use-কে **পরামর্শ** হিসেবে পাঠানো হয়।
   - পরের action Browser Use নিজেই বাছে।
5. **Recovery action-এর পর** P1 বিচার করে recovery কাজ করল কিনা।
6. **ঘটনা শেষ হলে** (ঠিক হয়ে গেলে, বা recovery-র সুযোগ শেষ হলে) trained **memory
   head** ঠিক করে অভিজ্ঞতাটা রাখার মতো কিনা। ঠিক হওয়া আর না-হওয়া, দুই ধরনের
   অভিজ্ঞতাই রাখা হয়, কারণ "এটা কাজ করেনি" জানাও দরকারি।
7. Task complete না হওয়া পর্যন্ত, অথবা ১৫ step শেষ না হওয়া পর্যন্ত এটা চলতে থাকে।

নিরপেক্ষতার জন্য সীমা: প্রতি ঘটনায় সর্বোচ্চ ২টা আর প্রতি episode-এ সর্বোচ্চ ৪টা
recovery; আর সব system-এর step ও model-call-এর বাজেট একই।

---

## ধাপ ৪ — কেন MiniWoB নয়, আসল website

প্রথমে আমরা পরীক্ষা করেছিলাম **MiniWoB**-এ, যেখানে ছোট ছোট খেলনা-ধরনের web page
থাকে। সেখানে P1 **১২২টা step-এর ১০০%-কে failure** বলেছিল, এমনকি যে task agent
complete করেছিল সেগুলোতেও। এটা bug কিনা সাবধানে যাচাই করা হয়েছে:

- আমাদের নিজের validation data-র ২৪০টা row হুবহু agent-এর code দিয়ে চালানো হয়েছে।
  Accuracy ০.৮২৫, MCC ০.৬৪৭ (Task 1-এর সমান)। অর্থাৎ code model-কে ঠিকভাবেই
  ব্যবহার করছে।
- Screenshot-এর সাইজ, website-এর নাম বা task-এর লেখা বদলেও ফল বদলায়নি।

আসল কারণ হলো model আমাদের dataset থেকে কী শিখেছে। আমাদের data-তে **সফল step-এ
সাধারণত page অনেকটা বদলে যায়** (median ৩৬% pixel), কারণ আসল site-এ ঠিক click
করলে প্রায়ই নতুন page খোলে। ব্যর্থ step-এ page প্রায় বদলায় না (median ০.৭%)।
MiniWoB-এ checkbox tick করলে মাত্র প্রায় ০.২% pixel বদলায়, তাই model প্রতিটা
MiniWoB step-কেই failure ভাবে।

**সমাধান:** agent-কে আমাদের dataset-এর আসল website-এ চালানো, training-এর সাইজে
(১২৮০×৭২০) screenshot দিয়ে। সেখানে P1 ঠিকভাবে বিচার করে:

| Step | P1 SUCCESS বলেছে |
|---|---:|
| MiniWoB, যেকোনো step | ০% |
| আসল site, **সফল** task-এর step | **৫৭%** (median P(failure) ০.৩৪) |
| আসল site, **ব্যর্থ** task-এর step | ১৮% (median P(failure) ১.০০) |

বিস্তারিত: [TASK2_WEB_SUITE.md](TASK2_WEB_SUITE.md)।

---

## ধাপ ৫ — Memory (P4) কীভাবে নতুন করে বানানো হলো

প্রথম memory কখনো কাজে আসতে পারত না। এর ১,৯৭৪টা উদাহরণে **element-এর তথ্য বা
সংশোধনের মান (value) ছিল না**, ৯৩% উদাহরণ নিয়মের কারণে সবসময় বাদ পড়ত, এটা অন্য
একটা পুরনো model-এর embedding-এ বানানো ছিল, আর agent এতে কিছু লিখতে পারত না।

নতুন **experience memory** agent-এর নিজের ঘটনাগুলো রাখে: কোন action ব্যর্থ হলো আর
কোন element-এ (তথ্যটা Browser Use থেকে নেওয়া), P1-এর diagnosis, প্রতিটা recovery
চেষ্টা আর তার ব্যাপারে P1-এর রায়, আর শেষে ঘটনাটা ঠিক হয়েছিল কিনা। খোঁজার চাবি
(key) আসে আমাদের Qwen2.5 model-এর memory embedding থেকে, আর কী রাখা হবে তা ঠিক
করে memory head।

একটা পরিমাপে দেখা গেছে শুধু memory embedding দিয়ে মিল থাকা অভিজ্ঞতা খোঁজা দুর্বল
(training data-তে AUC ০.৬০)। কারণ এটা শেখানো হয়েছিল *কী রাখা হবে* ঠিক করতে, মিল
খুঁজতে নয়। তাই খোঁজা এখন **একই web page**-এর মধ্যে সীমিত, embedding দিয়ে সাজানো,
আর শুধু সেই অভিজ্ঞতা দেখানো হয় যার element বর্তমান page-এ দেখা যায়।
বিস্তারিত: [P4_EXPERIENCE_MEMORY.md](P4_EXPERIENCE_MEMORY.md)।

---

## ধাপ ৬ — পথে যে সমস্যাগুলো পাওয়া গেছে আর ঠিক করা হয়েছে

| সমস্যা | সমাধান |
|---|---|
| Lab-এর PC hang হয়ে গিয়েছিল (memory leak: প্রতি model call-এ প্রায় ১২ GB আটকে থাকত) | প্রতিটা call-এর পর GPU memory ছেড়ে দেওয়া; free memory ১৬ GB-এর নিচে নামলে run থামে; ১২ GB-এর নিচে নামলে supervisor run বন্ধ করে দেয় |
| Actor ঠিক উত্তরকে Markdown code block-এ মুড়ে দিত | শুধু পুরো উত্তর ঘিরে থাকা block খোলা হয় (Browser Use নিজে যে নিয়ম মানে) |
| খালি action list-কে Browser Use-এর চেয়ে কঠোরভাবে বাতিল করা হচ্ছিল | Browser Use-এর নিজের retry নিয়মে ছেড়ে দেওয়া হয়েছে |
| কোনো error হলে run চুপচাপ আটকে যেত | Runner এখন প্রতিটা error দেখায় আর পরের task-এ চলে যায় |
| কিছু site robot আটকায় (DuckDuckGo) বা file download শুরু করে (Opera, Debian-এর button) | তুলনার আগেই সেই task বাদ দেওয়া বা নতুন করে লেখা হয়েছে |
| PC বন্ধ করায় একটা লম্বা run মারা গিয়েছিল | Run চলার সময় shutdown-lock থাকে; থামিয়ে আবার চালালে ফল হারায় না |

---

## ধাপ ৭ — আমাদের system সাহায্য করে কিনা, কীভাবে পরীক্ষা হবে

একই আসল website-এর task-এ দুটো system তুলনা করা হবে:

- **Baseline** = শুধু Browser Use।
- **Ours** = Browser Use + P1 recovery + P4 memory।

### Baseline যে task complete করে, সেটাও কেন ours দিয়ে চালানো হয়

1. **একই task সবসময় একই ফল দেয় না।** Wikipedia-র "Alan Turing" task একবার
   complete হয়েছিল, আরেকবার ব্যর্থ হয়েছে। কোন task "সহজ" তা আগে থেকে জানা যায় না।
2. **আমাদের system ক্ষতিও করতে পারে।** P1 কখনো কখনো ঠিক step-কেও failure বলে।
   তখন অপ্রয়োজনীয় recovery এমন task নষ্ট করতে পারে যেটা baseline পারত। এই
   ক্ষতিও মাপতে হবে।
3. **শুধু baseline-এর ব্যর্থ task বাছলে ফল পক্ষপাতদুষ্ট হবে।** একবার ব্যর্থ হওয়া
   task পরের বার এমনিতেই ভাগ্যক্রমে সফল হতে পারে। শুধু সেসব task-এ ours চালালে
   সেই ভাগ্যের সাফল্যও ভুলভাবে আমাদের system-এর কৃতিত্ব হিসেবে গোনা হবে।

### জোড়ায় জোড়ায় তুলনা (paired comparison)

প্রতিটা task দুই system দিয়ে একই দিনে পরপর চালানো হয়। কে আগে চলবে তা পালাক্রমে
বদলায়, যাতে website বদলালে দুই system-এর ওপর সমান প্রভাব পড়ে। প্রতিটা জোড়ার ফল
চারটার একটা:

| Baseline | Ours | মানে |
|---|---|---|
| ✅ | ✅ | সমান — দুটোই পেরেছে |
| ❌ | ❌ | সমান — কেউ পারেনি |
| ❌ | ✅ | **আমাদের system সাহায্য করেছে** |
| ✅ | ❌ | **আমাদের system ক্ষতি করেছে** |

তারপর "সাহায্য" আর "ক্ষতি" গুনে **exact two-sided sign test** করা হয়। আমাদের system
স্পষ্টভাবে ভালো বলা যাবে তখনই, যখন "সাহায্য" "ক্ষতি"-র চেয়ে অনেক বেশি হবে, যেমন
প্রায় ৭টা সাহায্য আর ০টা ক্ষতি, বা ১০টা সাহায্য আর ১টা ক্ষতি।

Task complete হলো কিনা তা ঠিক করে শুধু environment (page-এর URL task-এর লক্ষ্যের
সাথে মিললে)। Agent এই যাচাই কখনো দেখতে পায় না।

### কেন লাভ ছোট হতে পারে, আর কীভাবে পরিষ্কার করা যায়

আমাদের system সাহায্য করতে পারে শুধু তখন, যখন baseline ব্যর্থ হয় **এবং** ভুলটা
ঠিক করা সম্ভব **এবং** P1 ভুলটা ধরতে পারে **এবং** agent পরামর্শ মেনে চলে। ২০টা
task-এর development run-এ agent ১৯টার মধ্যে ১২টা complete করেছে। বাকি ৭টা ব্যর্থতার
৩টা ছিল actor-এর ভুল format-এর উত্তর, যা আমাদের system ঠিক করতে পারে না।

পার্থক্য স্পষ্ট করার উপায় (এগুলো চূড়ান্ত run-এর আগেই ঠিক করতে হবে, ফল দেখে নয়):
কঠিন একাধিক-ধাপের task নেওয়া, প্রতিটা task বেশিবার চালানো (যাতে আগের চেষ্টার
memory পরের চেষ্টায় কাজে লাগে), আর P1 খুব নিশ্চিত হলে তবেই recovery চালানো।

Task complete-এ বড় পার্থক্য না এলেও দুটো জিনিস বৈধ ফল হিসেবে থাকবে: আসল site-এ
P1 যে ঠিকভাবে failure ধরে তার প্রমাণ, আর যেসব ব্যর্থ task আমাদের system উদ্ধার
করেছে তার নির্দিষ্ট উদাহরণ।

---

## ধাপ ৮ — কীভাবে চালাবেন

```bash
cd /home/aiub/kiyas/webagent

# এক বা একাধিক task, আমাদের system দিয়ে বা ছাড়া; প্রতিটা step-এর বিবরণসহ
PYTHONPATH=src:scripts .task2-assets/browser-use-env/bin/python scripts/run_agent.py \
    --suite web --task wikipedia-wiki-alan-turing --mode ours

# Freeze করা baseline বনাম ours তুলনা (থামিয়ে আবার চালানো যায়)
PYTHONPATH=src:scripts .task2-assets/browser-use-env/bin/python -u scripts/compare_agents.py \
    run .task2-assets/comparison/web-v1
PYTHONPATH=src .venv/bin/python scripts/compare_agents.py report .task2-assets/comparison/web-v1

# আরেকটা terminal-এ live progress bar
python3 scripts/progress.py
```

পুরো নির্দেশনা: [TASK2_WEB_COMPARISON_RUNBOOK.md](TASK2_WEB_COMPARISON_RUNBOOK.md)।

---

## বর্তমান অবস্থা (২০২৬-০৯-২৮)

- Agent সম্পূর্ণ: Browser Use কাজ করছে, P1 বিচার করছে, P4 মনে রাখছে, আর সবকিছু
  আসল website-এ live চলছে।
- ২০টা task-এর development run: ১৯টার মধ্যে ১২টা complete; P1 ভালো আর খারাপ step
  আলাদা করতে পারছে।
- Baseline বনাম ours তুলনা (২০টা task × ২ বার × ২টা system = ৮০টা episode) freeze
  করা আর চালানোর জন্য তৈরি। ৬টা task-এর একটা pilot শুরু হয়েছে।
- পরের কাজ: তুলনাটা চালানো আর দেখা কোন system কতগুলো task complete করে।
