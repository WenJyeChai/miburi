# Forecast-Conditioned Causal Gesture Generation

## A mechanism-centered literature review for extending MIBURI

**Evidence cut-off:** 29 July 2026  
**Scope:** causal and streaming co-speech gesture generation; predicted lookahead; privileged-future distillation; speculative/receding-horizon generation; prospective and joint speech–gesture planning; psycholinguistic timing evidence.  
**Status convention:** “causal” below means that the current output uses no input sample or token timestamped after the current decision time. “Real-time” only means fast enough to run online. The two are not interchangeable.

---

## 1. Executive synthesis

### Bottom line

The specific research direction—**predict a distribution over near-future speech/intent from the causal prefix and use it to improve MIBURI’s current gesture decision without waiting for real future speech**—appears novel in the direct co-speech gesture literature as of the evidence cut-off.

The nearest gesture systems occupy different cells:

- [MIBURI](https://arxiv.org/abs/2603.03282) and [LiveGesture](https://arxiv.org/abs/2604.10927) are genuinely zero-lookahead and reactive.
- [STARGATE](https://doi.org/10.21437/Interspeech.2024-302) streams in chunks but gives each decision one second of **real future** audio and text.
- [Streaming Gesture Generation via Accelerated Rolling Diffusion](https://doi.org/10.1609/aaai.v40i31.39807) revises a speculative future motion window, but its current clean output is conditioned on an audio window extending into the **real future**.
- Semantic systems such as [Semantic Gesticulator](https://doi.org/10.1145/3658134), [RAG-Gesture](https://openaccess.thecvf.com/content/CVPR2025/html/Mughal_Retrieving_Semantics_from_the_Deep_an_RAG_Solution_for_Gesture_CVPR_2025_paper.html), and [EMAGE](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html) receive a complete utterance or window.
- Joint speech–gesture systems such as [Integrated Speech and Gesture Synthesis](https://arxiv.org/abs/2108.11436), [Match-TTSG](https://arxiv.org/abs/2310.05181), [FastTalker](https://doi.org/10.1016/j.neucom.2025.130074), and the 2025 [Gelina preprint](https://arxiv.org/abs/2510.12834) improve cross-modal coupling, but start from an available text script rather than a live, unfinished utterance.

The abstract mechanism is **not** new. [Alinejad, Siahbani, and Sarkar (EMNLP 2018)](https://aclanthology.org/D18-1337/) added a `PREDICT` action to simultaneous translation: a language model guessed unseen source words, the translation policy acted on those guesses, and predicted states were discarded when real input arrived. Action anticipation, privileged-future distillation, probabilistic trajectory forecasting, model-predictive control, and speculative decoding provide additional established mechanisms.

### What is scientifically plausible

There is a real timing problem to solve. Representational gesture onsets precede their lexical affiliates by about **370 ms** in a 1,803-gesture naturalistic corpus ([Donnellan et al., 2022](https://escholarship.org/uc/item/7w349725)); a separate conversational corpus found that a large majority of gestures began before the corresponding lexical information ([ter Bekke, Drijvers, & Holler, 2024](https://doi.org/10.1111/cogs.13407)). Classic production evidence also shows that lower word familiarity increases gesture–speech onset asynchrony ([Morrel-Samuels & Krauss, 1992](https://doi.org/10.1037/0278-7393.18.3.615)). This is compatible with a common conceptual plan in [de Ruiter’s Sketch Model](https://doi.org/10.1017/CBO9780511620850.018) and the interface account of [Kita and Özyürek (2003)](https://doi.org/10.1016/S0749-596X(02)00505-3).

It does **not** follow that every stroke should lead every word, or that a fixed 370–650 ms offset is correct. Preparation, stroke, hold, and retraction have different timing; gesture type, discourse function, lexical access, and interactional context change the relation. The model should therefore forecast **events and phases**, not merely shift all motion earlier.

### Recommended technical bet

The best first experiment is a **downstream, uncertainty-gated forecast head attached to the existing causal MIBURI pipeline**, not an immediate rewrite of Moshi:

1. From the same prefix available to MIBURI, predict a small multi-hypothesis plan for the next 0.16–1.0 s: speech act/entity relation, prosodic envelope, gesture event/type, and phase timing.
2. Feed posterior-weighted plan embeddings to the gesture transformer.
3. Permit low-risk preparation or beat motion under uncertainty, but begin an irreversible iconic/deictic stroke only when posterior confidence or hypothesis consensus is high.
4. Reforecast every 80 ms and invalidate stale hypotheses when real Moshi tokens disagree.
5. Compare against a matched-capacity auxiliary-loss control. This is essential because a deterministic forecast derived from the same prefix contains no new external information.

If that succeeds, the scientifically stronger but more invasive direction is a **shared prospective plan** generated before both speech and gesture. This better matches psycholinguistic theories and avoids independently sampling two modalities that later contradict one another.

### Main risks

- A forecast is a function of the same history and may add no information to a sufficiently expressive, optimally trained causal generator.
- The future is multimodal. A point estimate can turn uncertainty into confident but semantically wrong hand motion.
- Motion is partly irreversible. Speech tokens can be revised internally before emission; a visible stroke cannot be “unshown.”
- Existing metrics reward realism, diversity, and beat synchrony but barely measure semantic anticipation, phase timing, contradiction, calibration, or interruption behavior.
- Offline full-context teachers may transfer useful representations, but cannot distill information that is genuinely unpredictable from the prefix.

---

## 2. Review method and evidentiary rules

The first-stage map used MIBURI’s references and the peer-reviewed [Nyatsanga et al. co-speech gesture survey](https://doi.org/10.1111/cgf.14776), supplemented by recent action-anticipation survey terminology. The second stage followed backward/forward citation paths and searched primary work that claims causal, streaming, real-time, semantic, or joint speech–gesture generation, then traced the same information structure through action anticipation, human-motion and trajectory forecasting, world-model planning, simultaneous translation, streaming speech, and speculative decoding. Psycholinguistic evidence was included when it bears directly on gesture onset, lexical affiliates, phase timing, or common planning. Survey papers guided discovery but were not counted among the 45 annotated primary/foundational entries.

Primary proceedings, publisher pages, author-hosted papers, project pages, and official repositories were preferred. A preprint is labelled as such. If a paper says “real-time” but the retrieved methods do not establish its input-timestamp causality, the causal status is **unclear**, not inferred. Numeric horizon or latency is reported only where the primary source exposes it; otherwise it is “not reported” or “configuration-dependent.”

The search found no direct paper, through the cut-off date, that both:

1. predicts future semantic/acoustic/plan variables from a live causal speech prefix,
2. conditions the **current** co-speech gesture on that prediction,
3. consumes no real future speech at inference, and
4. explicitly manages forecast uncertainty or revision.

This is a bounded literature claim, not proof that no unpublished or poorly indexed work exists.

---

## 3. Information-flow taxonomy

Let \(C_t\) be all legitimately available context at decision time \(t\); \(X_{>t}\) the future external stream; \(\hat Z_{t:t+H}\) a forecast; and \(G_t\) the committed gesture output.

### 3.1 Reactive causal

\[
G_t \sim p_\theta(G_t \mid C_t)
\]

No future input is read and no explicit future variable is exposed. MIBURI and LiveGesture are the clearest direct examples. A causal model can still encode expectations implicitly in its hidden state; “reactive” here means no explicit forecast is represented or evaluated.

### 3.2 Bounded real lookahead

\[
G_t \sim p_\theta(G_t \mid C_t, X_{t:t+\Delta})
\]

The system waits \(\Delta\), then uses genuine future samples. This is low-latency but noncausal relative to time \(t\). STARGATE uses 1 s of future audio and text; Emformer and ANCAT expose the general accuracy–latency trade-off in streaming ASR.

### 3.3 Causal predicted lookahead

\[
\hat Z_{t:t+H} \sim q_\phi(Z_{t:t+H}\mid C_t),\qquad
G_t \sim p_\theta(G_t\mid C_t,\hat Z_{t:t+H})
\]

Only the prefix is externally observed. Alinejad et al.’s simultaneous-translation `PREDICT` action is the closest structural precedent; Vondrick et al., Gammulle et al., MultiPath, and VAP illustrate latent, task-oriented, multimodal, and joint-state forecasts.

### 3.4 Teacher-distilled future awareness

During training, a teacher sees future or complete context:

\[
h_t^{S}=f_S(C_t),\qquad h_t^{T}=f_T(C_t,X_{>t}),\qquad
\mathcal L_{\rm KD}=d(P h_t^S,h_t^T).
\]

At inference, only the student remains. This produces a causal student but not necessarily an explicit forecast. Tran et al. and PPKD are the key precedents. It is best treated as a separate baseline, because it changes representation learning without adding inference-time planning.

### 3.5 Speculative rollout / receding horizon

The system rolls out one or more possible futures, scores them, commits a short prefix, observes new evidence, and replans. PETS and PlaNet supply the control template; speculative decoding supplies accept/reject semantics; rolling diffusion supplies a temporally graded uncertain window. A gesture system needs stricter commitment logic because already rendered motion cannot be rolled back.

### 3.6 Shared prospective planner

\[
P_t \sim q(P_t\mid C_t),\quad
S_t \sim p(S_t\mid C_t,P_t),\quad
G_t \sim p(G_t\mid C_t,P_t).
\]

A common latent plan is made available before both modalities commit. MIBURI currently taps Moshi’s current internal speech/text streams, which is useful shared upstream state but not an explicit future plan. FastTalker’s TTS features and psycholinguistic production models motivate this category; a live causal implementation remains open.

### 3.7 Joint speech–gesture generation

\[
(S,G) \sim p_\theta(S,G\mid C)
\]

Integrated Speech and Gesture Synthesis, Match-TTSG, and Gelina model a joint distribution or interleaved stream. Joint generation and shared planning overlap but are not identical: a joint decoder can still receive a full script, and two separate decoders can still share an upstream causal plan.

**The categories are not mutually exclusive.** Rolling diffusion combines actual lookahead with speculative future motion; a shared planner can use multi-hypothesis predicted lookahead; a distilled student can also perform explicit forecasts. Every paper must therefore be classified by the timestamps it reads, not by architecture labels such as “autoregressive,” “streaming,” or “causal convolution.”

---

## 4. MIBURI as the reference system

[MIBURI: Towards Expressive Interactive Gesture Synthesis](https://openaccess.thecvf.com/content/CVPR2026/papers/Mughal_MIBURI_Towards_Expressive_Interactive_Gesture_Synthesis_CVPR_2026_paper.pdf) is a CVPR 2026 paper with an official [project page](https://vcai.mpi-inf.mpg.de/projects/MIBURI/) and [code repository](https://github.com/m-hamza-mughal/miburi).

Its relevant properties are:

- **Input:** causal semantic and acoustic token streams from the frozen 7B Moshi speech–text model, plus past gesture tokens.
- **Motion representation:** separate residual VQ codecs for upper body/hands, lower body/global translation, and FLAME facial parameters.
- **Generator:** a temporal transformer predicts the first residual code at a new time; a kinematic transformer fills residual/body-part codes at that time. Both temporal self-attention and speech/text cross-attention are causal.
- **Rate:** one gesture token represents two 25-fps frames, so a generation step covers 80 ms (12.5 Hz).
- **Context:** 25 gesture steps for causal self-attention and 50 speech/text steps for causal cross-attention.
- **Latency:** 34.9 ms per step on an A100 in the reported model comparison; approximately 36 ms per rendered frame on an RTX 3090 in the interactive setup; Moshi itself is reported at roughly 200 ms response latency. Hardware and measurement boundaries differ, so these numbers should not be added blindly.
- **Evaluation:** FGD, BeatAlign, L1 diversity, facial MSE, and human pairwise judgments of naturalness/alignment.
- **Observed causal trade-off:** the authors note that causal models tend toward beat gestures, while full-context semantic systems have an advantage for meaning-bearing gestures. Their supplement also recognizes that speech and gesture can precede or follow one another because both may arise from a common communicative intention.
- **Open limitations for this review:** no explicit future representation, forecast uncertainty, semantic anticipation metric, gesture-phase metric, plan consistency test, or user-body/dyadic conditioning.

MIBURI is therefore an unusually clean substrate for testing whether explicit forecast structure helps beyond a strong causal baseline. Its 80 ms clock also makes horizons operational: 2, 4, 8, 12, and 25 steps correspond to 0.16, 0.32, 0.64, 0.96, and 2.0 s.

---

## 5. Comparison matrix

**Reading rule:** “Strictly causal” is judged with respect to the external input stream named by the task. `Unclear` means the retrieved primary source did not establish the claim; `N/A` is reserved for human studies/theories without model inference. A complete text script is future information for a live unfinished dialogue turn even if the network that consumes it uses causal convolutions or autoregressive output.

The required fields are split into three tier tables only for readability; the columns are identical.

### 5.1 Tier A — Direct gesture and joint speech–gesture systems

| Paper | Domain | Strictly causal | Real lookahead | Predicted future | Future only during training | Shared plan | Joint outputs | Uncertainty | Horizon | Latency reported | Code available | Transferable idea |
|---|---|---:|---|---|---:|---|---|---|---|---|---:|---|
| [MIBURI 2026](https://arxiv.org/abs/2603.03282) | Live gesture + face | **Yes** | None | No explicit | No | No explicit; shared current Moshi state | No | Token sampling only | No future; 80 ms step | 34.9 ms/step A100; ≈36 ms rendered frame RTX 3090 | [Yes](https://github.com/m-hamza-mughal/miburi) | Reference causal backbone |
| [LiveGesture 2026](https://arxiv.org/abs/2604.10927) | Live full-body gesture | **Yes** | None | No | No | No | No | Training masks; not calibrated | Zero lookahead | Streaming established; comparable number unclear | Unclear | Independent reactive-causal baseline |
| [STARGATE 2024](https://doi.org/10.21437/Interspeech.2024-302) | Chunked gesture | **No** | **1 s audio + text** | No | No | No | No | None explicit | 1 s | Up to 7.5× real-time throughput | Unclear | Bounded actual-lookahead oracle |
| [Accelerated Rolling Diffusion 2026](https://doi.org/10.1609/aaai.v40i31.39807) | Streaming gesture diffusion | **No** | Real future audio across window | Yes—future motion window | No | No | No | Increasing diffusion noise with distance; not calibrated semantics | Rolling window \(N\); seconds unclear | Up to 200 fps / 4×; fastest setting ≈0.002 s | [Yes](https://github.com/andrewbo29/co-speech-gestures-rolling-diffusion) | Revise uncommitted future, then commit prefix |
| [EMAGE 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html) | Holistic gesture/face | **No** | Full sequence | No unseen-speech forecast | No | No | Body/face regions, not speech+gesture | Generative masking | Clip | No online claim | [Yes](https://pantomatrix.github.io/EMAGE/) | Full-context teacher/oracle |
| [ConvoFusion 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Mughal_ConvoFusion_Multi-Modal_Conversational_Diffusion_for_Co-Speech_Gesture_Synthesis_CVPR_2024_paper.html) | Dyadic gesture | **No** | Full clips | No | No | No | No | Diffusion samples/modality guidance | Clip | No online claim | Unclear | Partner-conditioned teacher |
| [RAG-Gesture 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Mughal_Retrieving_Semantics_from_the_Deep_an_RAG_Solution_for_Gesture_CVPR_2025_paper.html) | Semantic gesture | **No** | Full utterance/window | No | No | Retrieved exemplar, not shared speech plan | No | Retrieval/diffusion scores, uncalibrated | Gesture/clip | No online claim | Project available; code status unclear | Interpretable semantic forecast vocabulary |
| [Semantic Gesticulator 2024](https://doi.org/10.1145/3658134) | Semantic + rhythmic gesture | **No** | Full text/audio | No | No | Candidate semantic plan for motion only | No | Candidate sampling, uncalibrated | Utterance/clip | No online claim | [Yes](https://pku-mocca.github.io/Semantic-Gesticulator-Page/) | Forecast event/type rather than waveform |
| [MambaTalk 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/23c9c94227f937cfb50592a15e7fbb63-Abstract-Conference.html) | Full-body gesture | **No** | Full speech cross-attention | No | No | No | No | Token sampling | Clip | Fast sampling; not live-causality evidence | [Yes](https://kkakkkka.github.io/MambaTalk/) | Internal AR/SSM causality is insufficient |
| [GestureLSM 2025](https://arxiv.org/abs/2501.18898) | Gesture flow matching | **No** | Full speech window | No | No | No | No | Flow samples | Clip | RT sampling; MIBURI setup 144.7 ms/step | [Yes](https://andypinxinliu.github.io/GestureLSM) | Negative causalization result |
| [Audio2Photoreal 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ng_From_Audio_to_Photoreal_Embodiment_Synthesizing_Humans_in_Conversations_CVPR_2024_paper.html) | Dyadic body/face | **No** | Full clips | No | No | Partner context, no prospective plan | Body + face only | Diffusion samples | Clip | No online claim | [Yes](https://github.com/facebookresearch/audio2photoreal) | Rich dyadic teacher/data |
| [FastTalker 2025](https://doi.org/10.1016/j.neucom.2025.130074) | Text→speech + gesture | **No** for unfinished live text | Complete script | Prosody/duration predicted from known text | No | **Partial:** shared TTS features | **Yes** | No calibrated plan posterior | Utterance | 0.17 s per output second, RTX 3090 | Unclear | Share upstream prosodic plan |
| [Integrated Speech & Gesture 2021](https://arxiv.org/abs/2108.11436) | Text→speech + gesture | **No** for unfinished live text | Complete script | No unseen-text forecast | No | Joint latent/output process | **Yes** | Generative; uncalibrated | Utterance | Faster parallel variant; no live claim | [Yes](https://swatsw.github.io/isg_icmi21/) | Joint-distribution baseline |
| [Match-TTSG 2024](https://arxiv.org/abs/2310.05181) | Joint speech–gesture flow | **No** for unfinished live text | Complete script | No unseen-text forecast | No | Joint flow state | **Yes** | Flow samples | Utterance | ≈15× speedup over cited diffusion predecessor | [Yes](https://shivammehta25.github.io/Match-TTSG/) | Joint consistency |
| [Gelina 2025](https://arxiv.org/abs/2510.12834), preprint | Interleaved speech–gesture | **No** for unfinished live text | Complete script | Later output tokens, not unseen input | No | Unified token stream | **Yes** | Sampling; uncalibrated | Gesture every 15 speech tokens | Deployment latency unclear | [Yes](https://tguichoux.github.io/) | Interleave plan/speech/gesture tokens |

### 5.2 Tier B — Mechanistically adjacent work

| Paper | Domain | Strictly causal | Real lookahead | Predicted future | Future only during training | Shared plan | Joint outputs | Uncertainty | Horizon | Latency reported | Code available | Transferable idea |
|---|---|---:|---|---|---:|---|---|---|---|---|---:|---|
| [de Ruiter 2000](https://doi.org/10.1017/CBO9780511620850.018) | Psycholinguistic production | N/A—human theory | N/A | Prospective plan, not ML forecast | N/A | **Yes** | Speech + gesture production theory | Not formalized | Phrase/gesture | N/A | No | Common communicative plan |
| [Kita & Özyürek 2003](https://doi.org/10.1016/S0749-596X(02)00505-3) | Speech–gesture planning | N/A—human study | N/A | Conceptual planning | N/A | **Yes** | Speech + gesture evidence | Not formalized | Event/utterance | N/A | No | Semantic interface representation |
| [Morrel-Samuels & Krauss 1992](https://doi.org/10.1037/0278-7393.18.3.615) | Gesture timing | N/A—human study | N/A | No | N/A | Evidence for coordination | Observed modalities | Empirical variability | Subsecond | N/A | No | Confidence/lexical difficulty should affect lead |
| [Donnellan et al. 2022](https://escholarship.org/uc/item/7w349725) | Gesture timing | N/A—corpus study | N/A | No | N/A | Evidence for coordination | Observed modalities | Timing distribution | ≈370 ms mean onset lead | N/A | No | Choose short phase-aware horizons |
| [ter Bekke et al. 2024](https://doi.org/10.1111/cogs.13407) | Gesture predictive potential | N/A—corpus study | N/A | No computational forecast | N/A | Evidence for coordination | Observed modalities | Broad timing distribution | Subsecond | N/A | [Yes—data/scripts](https://osf.io/f9qm6/) | Earlier gesture is not automatically more useful |
| [Vondrick et al. 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Vondrick_Anticipating_Visual_Representations_CVPR_2016_paper.html) | Visual anticipation | **Yes** | None | **Yes** | No | No | No | Uncalibrated regression/classification | 1 s actions; 5 s objects | No deployment claim | Unclear | Predict task-relevant latent, not pixels |
| [RULSTM 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Furnari_What_Would_You_Expect_Anticipating_Egocentric_Actions_With_Rolling-Unrolling_LSTMs_ICCV_2019_paper.html) | Action anticipation | **Yes** | None | **Yes** | No | No | No | Class probabilities + modality attention | Benchmark-dependent | No deployment claim | Unclear | Multi-scale rolling/unrolling forecasts |
| [Gammulle et al. 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gammulle_Predicting_the_Future_A_Jointly_Learnt_Model_for_Action_Anticipation_ICCV_2019_paper.pdf) | Action anticipation | **Yes** | None | **Yes** | No | No | No | GAN diversity, uncalibrated | Dataset-dependent | No deployment claim | Unclear | Optimize forecast for downstream use |
| [Tran et al. 2021](https://arxiv.org/abs/1904.04868) | Anticipation distillation | **Yes—student** | None | No explicit at inference | **Yes** | No | No | No calibrated uncertainty | Remainder of action at training | No deployment claim | Unclear | Full-action teacher → causal student |
| [PPKD 2022](https://doi.org/10.1016/j.patcog.2022.108741) | Online action detection | **Yes—student** | None | No explicit at inference | **Yes** | No | No | No | Progressive future horizons in teachers | No deployment claim | Unclear | Partial-feature, progressive-horizon distillation |
| [FUTR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Gong_Future_Transformer_for_Long-Term_Action_Anticipation_CVPR_2022_paper.html) | Long action anticipation | **Yes** | None | **Yes** | No | No | No | Token class distributions, uncalibrated | Minutes | Parallel decoding; not RT benchmark | Project; code unclear | Parallel event-token future |
| [RAFTformer 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Girase_Latency_Matters_Real-Time_Action_Forecasting_Transformer_CVPR_2023_paper.html) | Real-time action forecasting | **Yes** | None | **Yes** | No | No | No | Class probabilities, uncalibrated | Dataset-defined | 9× lower latency at matched accuracy | Unclear | Charge inference time to forecast value |
| [DLow 2020](https://arxiv.org/abs/2003.08386) | Human-motion prediction | **Yes** | None | **Yes** | No | No | No | Diverse correlated samples; no mode probabilities | Dataset-defined | Efficient sampling; no RT claim | [Yes](https://www.ye-yuan.com/dlow) | Do not collapse to one mean future |
| [MultiPath 2019](https://arxiv.org/abs/1910.05449) | Driving trajectory forecast | **Yes** | None | **Yes** | No | No | No | Mode probabilities + Gaussian waypoint uncertainty | Multi-second | One forward pass | Unclear | Probabilistic anchors + residuals |
| [Trajectron++ 2020](https://arxiv.org/abs/2001.03093) | Multi-agent forecast | **Yes** | None | **Yes** | No | Optional ego plan conditions predictions | No | Graph-CVAE modes | Dataset-defined multi-second | Online integration intended | [Yes](https://github.com/StanfordASL/Trajectron-plus-plus) | Co-condition forecast and own plan |
| [CPC 2018](https://arxiv.org/abs/1807.03748), preprint | Predictive representation | **Yes—encoder** | None | No required at inference | **Yes** | No | No | Contrastive density ratio, not forecast calibration | Task-defined | Efficient objective | No canonical code | Auxiliary future-supervision control |
| [PETS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) | Model-based control | **Yes** | None | **Yes—rollouts** | No | Control plan + model rollout | No | Ensemble epistemic + probabilistic aleatoric | Task-specific MPC | Online control; compute-dependent | [Yes](https://github.com/kchua/handful-of-trials) | First-action commit and replan |
| [PlaNet 2019](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) | Latent planning | **Yes** | None | **Yes—latent rollouts** | No | Control plan + latent model | No | Stochastic latent samples | Finite CEM horizon | Online control; task-specific | [Yes](https://planetrl.github.io/) | Plan in compact latent space |
| [Legible Motion 2013](https://doi.org/10.15607/RSS.2013.IX.024) | Human-aware robot motion | Yes given known goal | None | No—planned trajectory | No | **Known goal plan** | No | Bayesian observer belief | Whole motion | Task-specific | Unclear | Gesture should make plan legible |

### 5.3 Tier C — Structurally analogous work

| Paper | Domain | Strictly causal | Real lookahead | Predicted future | Future only during training | Shared plan | Joint outputs | Uncertainty | Horizon | Latency reported | Code available | Transferable idea |
|---|---|---:|---|---|---:|---|---|---|---|---|---:|---|
| [Gu et al. 2017](https://aclanthology.org/E17-1099/) | Simultaneous MT | **Yes** | None; policy may wait for next real token | No | No | No | No | RL READ/WRITE policy | Adaptive lag | Quality–delay objective | ACL resources | Learn when to wait |
| [Alinejad et al. 2018](https://aclanthology.org/D18-1337/) | Simultaneous MT | **Yes** | None | **Yes—source words** | No | Predicted source state coordinates translation | No | LM distribution + policy | One/chained words | Better BLEU and delay reported | [Yes—attachment](https://aclanthology.org/D18-1337/) | Predict/use/discard when reality arrives |
| [STACL / wait-k 2019](https://aclanthology.org/P19-1289/) | Simultaneous MT | **Yes** | Real prefix obtained by waiting | No explicit | No | No | No | Token distribution | Fixed \(k\) words | Explicit latency control | ACL resources | Equal-delay wait baseline |
| [Monotonic Multihead Attention 2020](https://openreview.net/forum?id=Hyg96gBKPS) | Simultaneous MT | **Yes** | None | No | No | No | No | Monotonic boundary decisions | Adaptive lag | Latency losses/results | Links via OpenReview | Learned wait/commit gate |
| [Emformer 2021](https://arxiv.org/abs/2010.10759) | Streaming ASR | **No** | Configured right context | No | No | No | No | Decoder probabilities | Config-dependent | 80 and 960 ms average-latency settings | Unclear | Actual-lookahead oracle architecture |
| [ANCAT 2023](https://arxiv.org/abs/2305.04159) | Streaming ASR | **No** | **Adaptive real future audio** | No | No | No | No | Learned adaptive context | Variable | Explicit accuracy–latency results | Unclear | Use lookahead only when valuable |
| [Cascaded Encoders 2021](https://arxiv.org/abs/2010.14606) | Streaming/offline ASR | **Yes streaming; no offline branch** | Full utterance in second branch | No | No—separate inference modes | Shared streaming backbone | No | No explicit | Full utterance offline | Streaming retained; offline WER improves 10–27% relative | Unclear | Shared-backbone teacher layout |
| [VAP 2022](https://arxiv.org/abs/2205.09812) | Turn-taking forecast | **Yes** | None | **Yes—joint voice activity** | No | Joint future of both speakers | No output generation | Full 256-state probability | 2 s in 0.2/0.4/0.6/0.8 s bins | Later CPU real-time demo | [Yes](https://erikekstedt.github.io/VAP/) | Discrete joint future + calibration |
| [CosyVoice 2 2024](https://arxiv.org/abs/2412.10117), preprint | Streaming TTS | **Unclear** for unfinished text; causal relative to received chunks | Text may be preavailable | Speech tokens, not unseen text | No | No explicit | Speech only | Sampling | Chunk-configured | Minimal latency claimed; setup-dependent | [Yes](https://github.com/QwenAudio/CosyVoice) | Practical causal speech decoder |
| [Hibiki 2025](https://openreview.net/forum?id=fgjN8B6xVX) | Simultaneous speech translation | **Yes** external source | None; adaptive waiting | No unseen-source forecast | Training uses delay labels, not future teacher | Shared multistream state | **Target text + audio** | Sampling/perplexity-derived delay | Adaptive word delay | Real-time/on-device reported | Yes, linked by authors | Joint streams + learned semantic delay |
| [Speculative Decoding 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | AR inference | **Yes** | None | **Yes—draft tokens** | No | Draft verified by target | No | Exact accept/reject probability | \(K\) tokens | 2–3× speedup | Unclear | Keep drafts uncommitted until verified |

---

## 6. Annotated bibliography

The annotations emphasize what can actually be transferred. “Future” distinguishes a measured future signal from a model prediction. Latency figures use each paper’s own boundary and hardware and are therefore not directly rankable.

### 6.1 Direct work: causal, streaming, semantic, and joint gesture generation

#### D1. Mughal et al. — MIBURI (CVPR 2026, published)

**Source:** [paper](https://arxiv.org/abs/2603.03282), [project](https://vcai.mpi-inf.mpg.de/projects/MIBURI/), [code](https://github.com/m-hamza-mughal/miburi).  
**Problem and inputs:** expressive full-body gesture and facial generation during live Moshi dialogue, using past gesture tokens and current/past Moshi semantic/acoustic states. **Causality/future:** strictly causal masks; no real or predicted future. **Representation/horizon/uncertainty:** three regional RVQ codecs, 80 ms motion tokens, 25-step motion history and 50-step speech/text history; stochastic token generation but no forecast posterior. **Latency/result:** 34.9 ms/step on A100; reported FGD 0.480 with face in the multi-speaker evaluation, while remaining causal and real-time. **Limitations/relevance:** tends toward beat gestures relative to semantic full-context systems and lacks phase/semantic anticipation evaluation. It is the exact backbone and control condition.

#### D2. Ma et al. — LiveGesture (CVPR 2026, published)

**Source:** [paper](https://arxiv.org/abs/2604.10927), [official CVPR index](https://openaccess.thecvf.com/CVPR2026?day=2026-06-05).  
**Problem and inputs:** arbitrary-length, audio-driven full-body motion from a causal streamable audio encoder and prior region tokens. **Causality/future:** explicitly zero lookahead; neither real nor predicted future is used. **Representation/horizon/uncertainty:** regional VQ tokenization and hierarchical autoregression; random regional masking improves training robustness but is not a future-confidence mechanism. **Latency/result:** the paper establishes streaming operation and competitive quality; a like-for-like latency boundary with MIBURI is not available in the retrieved record. **Limitations/relevance:** no explicit semantic plan or forecast. It is the most important independent reactive-causal baseline.

#### D3. Abel et al. — STARGATE (Interspeech 2024, published)

**Source:** [paper](https://www.isca-archive.org/interspeech_2024/abel24_interspeech.pdf), [DOI](https://doi.org/10.21437/Interspeech.2024-302).  
**Problem and inputs:** fast chunked autoregressive gesture synthesis from audio, text, and motion history. **Causality/future:** each prediction uses one second of past **and one second of real future** audio/text; it is streaming but noncausal. **Representation/horizon/uncertainty:** continuous motion chunks processed with modified bidirectional GRUs; no predicted-future distribution or calibration. **Latency/result:** up to 7.5× faster than real time on long sequences while maintaining quality. **Limitations/relevance:** speed claims must not be converted into causality claims. It is the right 1 s actual-lookahead oracle against which a forecasted 1 s condition can be compared.

#### D4. Vu, Boiarov, and Vetrov — Accelerated Rolling Diffusion (AAAI 2026, published)

**Source:** [paper](https://ojs.aaai.org/index.php/AAAI/article/view/39807), [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39807/43768), [code](https://github.com/andrewbo29/co-speech-gestures-rolling-diffusion).  
**Problem and inputs:** convert diffusion gesture models into continuous rolling generators without stitched post-processing. **Causality/future:** the rolling motion future is speculative, but the conditioning audio window extends beyond the current output; this is not zero-lookahead. **Representation/horizon/uncertainty:** a window of future poses receives progressively greater diffusion noise; horizon is a model window \(N\), with no single seconds value established here. **Latency/result:** up to 200 fps and 4× acceleration; the fastest reported setting is about 0.002 s. **Limitations/relevance:** temporal noise is not calibrated semantic uncertainty. The valuable transfer is revise-until-commit scheduling, not its causal status.

#### D5. Liu et al. — EMAGE (CVPR 2024, published)

**Source:** [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html), [project/code](https://pantomatrix.github.io/EMAGE/).  
**Problem and inputs:** unified facial, body, hand, and lower-body generation from full speech/text and optional masked motion hints. **Causality/future:** full-window masked generation is noncausal for live use. **Representation/horizon/uncertainty:** four compositional VQ-VAEs and a masked audio-gesture transformer; generative inpainting covers the clip but does not expose forecast calibration. **Result:** strong holistic generation and a widely used BEAT2 baseline. **Limitations/relevance:** cannot establish forecast value because it reads the answer. It is an excellent full-context teacher and upper-bound quality reference.

#### D6. Mughal et al. — ConvoFusion (CVPR 2024, published)

**Source:** [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Mughal_ConvoFusion_Multi-Modal_Conversational_Diffusion_for_Co-Speech_Gesture_Synthesis_CVPR_2024_paper.html), [project](https://vcai.mpi-inf.mpg.de/projects/ConvoFusion/).  
**Problem and inputs:** multimodal, dyadic conversational gesture conditioned on speaker audio/text and interlocutor context. **Causality/future:** full clips are available to diffusion; noncausal. **Representation/horizon/uncertainty:** continuous clip-level diffusion with modality guidance and word-excitation control; sample diversity is not a calibrated forecast posterior. **Result:** demonstrates that partner context and separately steerable modalities improve conversational appropriateness. **Limitations/relevance:** costly whole-window generation and no online commitment. It motivates adding user/partner forecast variables after the one-speaker mechanism is validated.

#### D7. Mughal et al. — RAG-Gesture (CVPR 2025, published)

**Source:** [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Mughal_Retrieving_Semantics_from_the_Deep_an_RAG_Solution_for_Gesture_CVPR_2025_paper.html), [project](https://vcai.mpi-inf.mpg.de/projects/RAG-Gesture/).  
**Problem and inputs:** improve rare semantic gestures by retrieving annotated gesture exemplars from full text and guiding diffusion through DDIM inversion. **Causality/future:** full utterance/window; no causal future prediction. **Representation/horizon/uncertainty:** interpretable semantic exemplar plus motion latent, covering the retrieved gesture/clip; retrieval and diffusion scores are not calibrated for commit decisions. **Result:** stronger semantic controllability and long-tail behavior. **Limitations/relevance:** retrieval assumes semantic evidence already exists in the text. Its gesture vocabulary is a practical forecast target and its full-context model a teacher.

#### D8. Ao et al. — Semantic Gesticulator (ACM TOG 2024 / SIGGRAPH, published)

**Source:** [paper](https://arxiv.org/abs/2405.09814), [DOI](https://doi.org/10.1145/3658134), [project/code/data](https://pku-mocca.github.io/Semantic-Gesticulator-Page/).  
**Problem and inputs:** combine meaningful semantic gestures with rhythmic audio-driven motion. **Causality/future:** an LLM uses full textual context to retrieve from roughly 200 semantic gesture categories; generation is noncausal for live speech. **Representation/horizon/uncertainty:** discrete gesture candidates and a GPT-style audio-motion model over the utterance; no confidence-based action gate. **Result:** improves semantic gesture realization and offers an interpretable library. **Limitations/relevance:** LLM retrieval quality and library coverage cap behavior. Its type inventory is better suited to prediction than raw future waveform.

#### D9. Xu et al. — MambaTalk (NeurIPS 2024, published)

**Source:** [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/23c9c94227f937cfb50592a15e7fbb63-Abstract-Conference.html), [project/code](https://kkakkkka.github.io/MambaTalk/).  
**Problem and inputs:** high-quality full-body gesture from multimodal speech using local/global discrete priors and Mamba sequence modeling. **Causality/future:** the state-space scan is causal internally, but published speech cross-attention uses the full conditioning sequence; input-time causality is noncausal. **Representation/horizon/uncertainty:** local/global VQ motion tokens over a clip; sampling without calibrated future risk. **Result:** fast, strong offline generation. **Limitations/relevance:** a textbook example of why “Mamba/autoregressive” does not establish causal conditioning; useful as a matched architectural baseline after applying explicit masks.

#### D10. Liu et al. — GestureLSM (ICCV 2025, published)

**Source:** [paper](https://arxiv.org/abs/2501.18898), [project/code](https://andypinxinliu.github.io/GestureLSM).  
**Problem and inputs:** accelerate high-quality co-speech motion with latent shortcut flow matching and spatial–temporal region modeling. **Causality/future:** uses full speech windows; fast but noncausal. **Representation/horizon/uncertainty:** continuous regional motion latent and clip-level flow samples; no explicit forecast posterior. **Result:** strong offline FGD and real-time sampling. **Limitations/relevance:** MIBURI reports that a causalized GestureLSM changes FGD from 0.537 to 2.792 in its evaluation, demonstrating that masking an offline model is not enough. This is critical negative evidence for purpose-built causal planning.

#### D11. Ng et al. — Audio2Photoreal (CVPR 2024, published)

**Source:** [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ng_From_Audio_to_Photoreal_Embodiment_Synthesizing_Humans_in_Conversations_CVPR_2024_paper.html), [code/data](https://github.com/facebookresearch/audio2photoreal).  
**Problem and inputs:** generate photorealistic conversational body and facial motion from dyadic audio. **Causality/future:** diffusion receives full clips and partner context; noncausal. **Representation/horizon/uncertainty:** separate face/body latent diffusion over clips; multiple samples but no action-confidence semantics. **Result:** compelling dyadic motion and a valuable large-scale data/resource release. **Limitations/relevance:** offline and high-capacity. It is most useful as a full-context teacher and later partner-conditioned extension.

#### D12. Guo and Zhang — FastTalker (Neurocomputing 2025, published)

**Source:** [paper/DOI](https://doi.org/10.1016/j.neucom.2025.130074), [arXiv version](https://arxiv.org/abs/2409.16404).  
**Problem and inputs:** jointly synthesize speech waveforms and full-body gestures from text while avoiding a serial TTS→gesture pipeline. **Causality/future:** the authors causalize internal networks, but the complete text script is available, so it is not causal with respect to an unfinished live utterance. **Representation/horizon/uncertainty:** TTS duration, pitch, onset, and energy features are shared with the gesture branch over the script; no calibrated alternatives. **Latency/result:** 0.17 s computation per second of output on an RTX 3090 and strong BEAT2 results. **Limitations/relevance:** code availability is unclear and script access hides semantic uncertainty. The feature-sharing pattern is directly reusable with a causal plan forecaster.

#### D13. Kucherenko et al. — Integrated Speech and Gesture Synthesis (ICMI 2021, published)

**Source:** [paper](https://arxiv.org/abs/2108.11436), [project/code](https://swatsw.github.io/isg_icmi21/).  
**Problem and inputs:** learn speech acoustics and 3D gestures jointly from text, with Tacotron-like autoregressive and Glow-TTS-like parallel variants. **Causality/future:** complete text is supplied; output autoregression does not make the live-input problem causal. **Representation/horizon/uncertainty:** joint utterance-level acoustic and skeleton outputs; no explicit multi-hypothesis plan. **Result:** fewer parameters and better cross-modal behavior than separated pipelines in its evaluations. **Limitations/relevance:** small-data and scripted-input assumptions. It establishes that joint modeling can improve consistency.

#### D14. Mehta et al. — Match-TTSG (ICASSP 2024, published)

**Source:** [paper](https://arxiv.org/abs/2310.05181), [project/code](https://shivammehta25.github.io/Match-TTSG/).  
**Problem and inputs:** synthesize speech acoustics and skeleton gesture together from full text using optimal-transport conditional flow matching. **Causality/future:** full script; noncausal for an unfinished dialogue turn. **Representation/horizon/uncertainty:** one joint continuous trajectory over the utterance; flow samples represent variability but no semantic confidence gate. **Latency/result:** substantially fewer network evaluations and about a 15× speedup over the cited diffusion predecessor, with better subjective speech/gesture results. **Limitations/relevance:** no incremental plan revision. It is the cleanest joint continuous-distribution baseline.

#### D15. Guichoux et al. — Gelina (2025 preprint)

**Source:** [preprint](https://arxiv.org/abs/2510.12834), [demo/code](https://tguichoux.github.io/).  
**Problem and inputs:** jointly generate speech and co-speech gesture from text using one interleaved autoregressive stream; also supports gesture-only generation from speech. **Causality/future:** generation is causal in output-token order, but joint mode receives complete text. **Representation/horizon/uncertainty:** 75 Hz speech tokens with a 5 Hz gesture token every 15 speech tokens, followed by modality-specific decoders; ordinary sampling, no calibrated shared-plan posterior. **Result:** reported competitive speech and improved gesture generation, multi-speaker/style cloning. **Limitations/relevance:** peer-reviewed venue was not verified by the cut-off and live unfinished-text behavior is untested. It is the clearest code template for an interleaved prospective plan token.

### 6.2 Adjacent work: timing, future supervision, uncertainty, and planning

#### A1. de Ruiter — The Production of Gesture and Speech (2000, scholarly book chapter)

**Source:** [chapter DOI](https://doi.org/10.1017/CBO9780511620850.018).  
**Problem/evidence:** explains how speech and representational gesture are planned and temporally coordinated. The Sketch Model places both under a shared communicative intention/conceptualizer, with separate formulation routes and coordination signals. **Causality/future/representation:** this is a cognitive architecture, not a numeric forecast or runtime system; its key representation is a prospective “gesture sketch” paired with a speech plan. **Limitations/relevance:** it does not prescribe a neural implementation or calibrated uncertainty. It is the strongest theoretical justification for a shared upstream plan rather than gesture prediction from already emitted acoustics alone.

#### A2. Kita and Özyürek — What Does Cross-Linguistic Variation in Semantic Coordination Reveal? (JML 2003, published)

**Source:** [paper DOI](https://doi.org/10.1016/S0749-596X(02)00505-3).  
**Problem/evidence:** cross-linguistic experiments test how language-specific packaging changes iconic gesture. Results support an interface representation where spatial thinking and linguistic formulation interact, with a communication planner selecting content and modality. **Causality/horizon/uncertainty:** human event/utterance planning; no computational posterior or latency. **Limitations/relevance:** theory and controlled data do not determine a model target directly. They argue for forecasting semantic event structure and communicative function, not just audio frames.

#### A3. Morrel-Samuels and Krauss — Word Familiarity Predicts Temporal Asynchrony (JEP:LMC 1992, published)

**Source:** [DOI](https://doi.org/10.1037/0278-7393.18.3.615).  
**Problem/evidence:** 17 speakers described photographs; 129 judges identified 60 gesture–lexical-affiliate pairs. Gesture onset preceded speech more when the affiliate was less familiar, and familiarity related systematically to gesture duration/asynchrony. **Representation/horizon:** onset, duration, and lexical-affiliate timing at subsecond scale. **Limitations/relevance:** small number of target gestures and elicited narrative task. It rules against a universal fixed lead and suggests that forecast confidence/lexical-access difficulty should modulate preparation timing.

#### A4. Donnellan et al. — Timing Relationships Between Representational Gestures and Speech (CogSci 2022, published proceedings)

**Source:** [paper and abstract](https://escholarship.org/uc/item/7w349725).  
**Problem/evidence:** tests whether naturalistic representational gestures align more closely with utterance onset or their lexical affiliates. Across 1,803 gestures from 24 speakers, lexical-affiliate timing was the tighter anchor and gestures preceded affiliates by roughly 370 ms on average. **Uncertainty:** the corpus exposes a distribution, not a deterministic rule. **Limitations/relevance:** conference paper and corpus-specific annotations; onset does not equal semantic stroke. It motivates 0.32–0.64 s forecast conditions and lexical-affiliate evaluation.

#### A5. ter Bekke, Drijvers, and Holler — Hand Gestures Have Predictive Potential During Conversation (Cognitive Science 2024, published)

**Source:** [paper](https://doi.org/10.1111/cogs.13407), [open record/PDF](https://pure.mpg.de/pubman/faces/ViewItemFullPage.jsp?itemId=item_3565972_3).  
**Problem/evidence:** annotates gesture onset, stroke, lexical affiliate, and conversational response timing to ask whether gestures provide advance semantic information. Most gestures began before the corresponding lexical material, with broad timing variability; gesture presence could help responses in some comparisons, but greater lead was not itself reliable evidence of greater facilitation. The authors provide [data and analysis scripts](https://osf.io/f9qm6/). **Limitations/relevance:** observational timing does not establish a causal production mechanism. This is the strongest warning that “earlier” is not automatically “better,” and it justifies phase-specific human tests.

#### A6. Vondrick, Pirsiavash, and Torralba — Anticipating Visual Representations (CVPR 2016, published)

**Source:** [official paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Vondrick_Anticipating_Visual_Representations_CVPR_2016_paper.html).  
**Problem and inputs:** predict useful future visual representations from an observed video prefix, then apply ordinary recognition heads. **Causality/future:** causal observations; predicted, never observed, future at inference. **Representation/horizon/uncertainty:** feature embeddings rather than pixels; roughly 1 s for actions and 5 s for objects; no calibrated multimodal posterior. **Result:** predicted features aid future recognition. **Limitations/relevance:** deterministic/regressive representations blur modes. It supports forecasting a semantic/prosodic latent tailored to gesture, not raw future speech.

#### A7. Furnari and Farinella — RULSTM (ICCV 2019, published)

**Source:** [official paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Furnari_What_Would_You_Expect_Anticipating_Egocentric_Actions_With_Rolling-Unrolling_LSTMs_ICCV_2019_paper.html).  
**Problem and inputs:** anticipate egocentric actions by rolling observed RGB/flow/object histories and unrolling recurrent state toward the future. **Causality/future:** only past frames at inference; future states/actions are predicted. **Representation/horizon/uncertainty:** multi-scale recurrent future state and class distributions, with adaptive modality attention; benchmark-specific anticipation horizons. **Result:** strong EPIC-style anticipation. **Limitations/relevance:** probabilities are not designed as calibrated act/abstain decisions. Its rolling/unrolling separation maps naturally to MIBURI history encoding plus short forecast decoding.

#### A8. Gammulle et al. — Predicting the Future: A Jointly Learnt Model for Action Anticipation (ICCV 2019, published)

**Source:** [official paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gammulle_Predicting_the_Future_A_Jointly_Learnt_Model_for_Action_Anticipation_ICCV_2019_paper.pdf).  
**Problem and inputs:** anticipate actions from partial video while forecasting visual and temporal future embeddings. **Causality/future:** causal partial observation; future embeddings are generated by a recurrent GAN. **Representation/horizon/uncertainty:** task-oriented future feature sequence, dataset-defined horizon; adversarial diversity without calibrated probability. **Result:** joint training outperformed then-current baselines across several datasets. **Limitations/relevance:** GAN plausibility does not guarantee downstream calibration. It motivates end-to-end forecast usefulness rather than optimizing forecast reconstruction alone.

#### A9. Tran et al. — Knowledge Distillation for Human Action Anticipation (ICIP 2021, published)

**Source:** [paper](https://arxiv.org/abs/1904.04868), [conference record](https://www2.securecms.com/ICIP2021/Papers/ViewPaper_MS.asp?PaperNum=2873).  
**Problem and inputs:** train a partial-observation action anticipator with an action-recognition teacher that sees the complete action. **Causality/future:** teacher future is training-only; student is causal at inference and emits no explicit future. **Representation/horizon/uncertainty:** semantic feature distillation with a positional-shift-tolerant loss; future spans the unobserved remainder. **Result:** improvements on JHMDB and EPIC-KITCHENS, including use of unlabeled data. **Limitations/relevance:** cannot transfer unpredictable details and does not expose confidence. This is the first teacher/student architecture to reproduce for gesture.

#### A10. Zhao et al. — Progressive Privileged Knowledge Distillation (Pattern Recognition 2022, published)

**Source:** [DOI/publisher record](https://doi.org/10.1016/j.patcog.2022.108741).  
**Problem and inputs:** improve online action detection using an offline teacher with complete video. **Causality/future:** causal student at inference; future frames are privileged training information. **Representation/horizon/uncertainty:** only selected hidden features are distilled; intermediate teachers expose progressively more future, creating an easy-to-hard curriculum. **Result:** 87.24% mcAP on TVSeries and 64.46% mAP on THUMOS14 in the paper, with larger early-action gains. **Limitations/relevance:** no explicit forecast or calibrated uncertainty. Partial-feature and progressive-horizon distillation directly address the impossibility of perfectly matching a future-seeing teacher.

#### A11. Gong et al. — FUTR (CVPR 2022, published)

**Source:** [official paper](https://openaccess.thecvf.com/content/CVPR2022/html/Gong_Future_Transformer_for_Long-Term_Action_Anticipation_CVPR_2022_paper.html).  
**Problem and inputs:** predict minutes-long sequences of future actions from observed video without autoregressive error accumulation. **Causality/future:** observed segment only; future action tokens are predicted. **Representation/horizon/uncertainty:** parallel categorical action labels/durations over minutes; token probabilities but no selective calibration. **Result:** state of the art on Breakfast and 50Salads at publication. **Limitations/relevance:** horizons and event granularity are much coarser than gesture. The transferable idea is parallel event-token prediction rather than dense raw-frame forecasting.

#### A12. Girase et al. — RAFTformer (CVPR 2023, published)

**Source:** [official paper](https://openaccess.thecvf.com/content/CVPR2023/html/Girase_Latency_Matters_Real-Time_Action_Forecasting_Transformer_CVPR_2023_paper.html), [project](https://karttikeya.github.io/publication/RAFTformer/).  
**Problem and inputs:** action forecasting whose score includes the time spent computing the answer. **Causality/future:** causal observed clips; future action is predicted. **Representation/horizon/uncertainty:** action-class distribution at dataset-defined horizon; shuffled causal masking aids generalization. **Latency/result:** 9× lower inference latency at comparable forecast accuracy, 94% less training compute, and improved real-time top-5 recall. **Limitations/relevance:** does not handle physical commitment. It supplies the correct evaluation principle: late forecasts have less—or negative—gesture value.

#### A13. Yuan and Kitani — DLow (ECCV 2020, published)

**Source:** [paper](https://arxiv.org/abs/2003.08386), [code](https://www.ye-yuan.com/dlow).  
**Problem and inputs:** efficiently draw a deliberately diverse set of plausible human futures from past motion. **Causality/future:** causal pose history; predicted future poses only. **Representation/horizon/uncertainty:** one latent is mapped to correlated codes and continuous trajectories, with a diversity-promoting prior; samples lack calibrated mode probabilities. **Result:** better diversity/accuracy than independent Gaussian sampling. **Limitations/relevance:** diversity can select implausible or semantically inconsistent candidates. It is evidence that one mean future is a poor gesture forecast.

#### A14. Chai et al. — MultiPath (CoRL 2019, published)

**Source:** [paper](https://arxiv.org/abs/1910.05449).  
**Problem and inputs:** predict multimodal driving trajectories from history and scene context in one efficient pass. **Causality/future:** past only; all future trajectories are predicted. **Representation/horizon/uncertainty:** fixed anchor trajectories, categorical mode probabilities, offsets, and per-waypoint Gaussian uncertainty over a multi-second horizon. **Result:** more accurate distributions with far fewer trajectories than sampling baselines. **Limitations/relevance:** fixed anchors need sufficient coverage and calibration can drift. This is the best simple design for \(K\) gesture-intent candidates plus probabilities and timing residuals.

#### A15. Salzmann et al. — Trajectron++ (ECCV 2020, published)

**Source:** [paper](https://arxiv.org/abs/2001.03093), [code/models/data](https://github.com/StanfordASL/Trajectron-plus-plus).  
**Problem and inputs:** dynamically feasible multi-agent trajectory forecasting using histories, interaction graphs, maps, and optionally the ego agent’s planned trajectory. **Causality/future:** historical input only; future trajectories are sampled. **Representation/horizon/uncertainty:** graph CVAE modes over dataset-specific multi-second horizons. **Result:** strong benchmark accuracy and planner integration. **Limitations/relevance:** CVAE likelihood and sample diversity are not automatically decision calibrated. The important transfer is bidirectional consistency: forecast others conditional on one’s own plan, not independently.

#### A16. van den Oord, Li, and Vinyals — Contrastive Predictive Coding (2018 preprint)

**Source:** [preprint](https://arxiv.org/abs/1807.03748).  
**Problem and inputs:** learn representations by predicting future latent features from an autoregressive past context. **Causality/future:** future samples are training targets; the deployed encoder can remain causal and need not output a forecast. **Representation/horizon/uncertainty:** latent embeddings and InfoNCE probability ratios at task-chosen offsets. **Result:** broadly useful features across speech, images, text, and RL. **Limitations/relevance:** contrastive predictability may emphasize easy nuisance signals and is not a calibrated generative forecast. It is the essential auxiliary-loss control for the “explicit forecast head” claim.

#### A17. Chua et al. — PETS (NeurIPS 2018, published)

**Source:** [official paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [code](https://github.com/kchua/handful-of-trials).  
**Problem and inputs:** sample-efficient model-based control with uncertainty-aware learned dynamics. **Causality/future:** current state/history only; future state particles are imagined. **Representation/horizon/uncertainty:** probabilistic ensemble separates useful epistemic and aleatoric effects; trajectories are scored over an MPC horizon. **Commit/revise:** apply the first action only, observe, and replan. **Result:** strong control with relatively few trials. **Limitations/relevance:** rollout compute can be high, and model error compounds. It provides the safest template for confidence-aware gesture commitment.

#### A18. Hafner et al. — PlaNet (ICML 2019, published)

**Source:** [paper](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf), [project](https://planetrl.github.io/).  
**Problem and inputs:** plan from pixels using stochastic latent dynamics rather than costly image rollouts. **Causality/future:** causal belief state; candidate latent futures are imagined. **Representation/horizon/uncertainty:** compact stochastic latent trajectories optimized by cross-entropy-method planning over a finite task-specific horizon. **Commit/revise:** execute the first action and replan. **Result:** strong visual-control data efficiency. **Limitations/relevance:** a learned reward is needed and planning latency matters. It motivates forecasted semantic/prosodic latents rather than waveform or dense-pose rollouts.

#### A19. Dragan, Lee, and Srinivasa — Legibility and Predictability of Robot Motion (RSS 2013, published)

**Source:** [DOI](https://doi.org/10.15607/RSS.2013.IX.024).  
**Problem and inputs:** design robot motion so a human observer infers the intended goal early, distinguishing legible from merely predictable motion. **Causality/future:** the robot already knows its goal and plans a trajectory; this is not future-input prediction. **Representation/horizon/uncertainty:** Bayesian observer belief over goals across a full trajectory. **Result:** legible paths can differ from efficient/predictable paths and improve early goal inference. **Limitations/relevance:** stylized tasks and known goals. It reframes gesture as an action that *communicates and commits to* a plan, so consistency matters more than raw forecast accuracy.

### 6.3 Analogous work: simultaneous output, adaptive waiting, and speculative verification

#### X1. Gu et al. — Learning to Translate in Real-Time with Neural MT (EACL 2017, published)

**Source:** [ACL Anthology](https://aclanthology.org/E17-1099/).  
**Problem and inputs:** learn when to READ another source token or WRITE a target token, balancing translation quality and latency. **Causality/future:** causal source prefix; no future prediction. **Representation/horizon/uncertainty:** reinforcement-learned READ/WRITE policy; variable lag. **Result:** learned policies improve the quality–delay trade-off over simple schedules. **Limitations/relevance:** output tokens are less physically irreversible than gestures. It is the necessary policy baseline: sometimes waiting is more rational than forecasting.

#### X2. Alinejad, Siahbani, and Sarkar — Prediction Improves Simultaneous NMT (EMNLP 2018, published)

**Source:** [paper and attachment](https://aclanthology.org/D18-1337/).  
**Problem and inputs:** improve simultaneous translation without waiting for unseen source words. **Causality/future:** a causal language model proposes the next source word; the agent chooses READ, WRITE, or PREDICT and can chain predictions. No true future input is read. **Representation/horizon/uncertainty:** discrete predicted words and their encoder hidden states; forecast probability affects policy reward, but modern calibration is absent. **Commit/revise:** predicted hidden states are discarded/reset when a real READ arrives; already emitted translation remains. **Result:** English→German BLEU rose 16.75→17.54 while average proportion fell .79→.74; German→English BLEU rose 21.43→21.83 while proportion fell .77→.70. Excess prediction hurt, and around 20% PREDICT actions was most useful. **Relevance:** this is the closest exact precedent and directly motivates selective, not constant, gesture forecasting.

#### X3. Ma et al. — STACL / wait-k (ACL 2019, published)

**Source:** [ACL Anthology](https://aclanthology.org/P19-1289/).  
**Problem and inputs:** prefix-to-prefix simultaneous translation with a simple fixed delay. **Causality/future:** waits for \(k\) source words, then stays \(k\) words behind; no predicted unseen source. **Representation/horizon/uncertainty:** ordinary target-token distribution; fixed lag \(k\). **Result:** controllable latency with reasonable quality across Chinese↔English and German↔English. **Limitations/relevance:** fixed lag ignores local uncertainty. A gesture study must include analogous real-lookahead/wait baselines; otherwise forecast gains may simply reflect extra effective delay.

#### X4. Ma et al. — Monotonic Multihead Attention (ICLR 2020, published)

**Source:** [OpenReview](https://openreview.net/forum?id=Hyg96gBKPS).  
**Problem and inputs:** learn incremental monotonic attention for simultaneous translation rather than hand-set read/write schedules. **Causality/future:** source prefix only; no unseen source prediction. **Representation/horizon/uncertainty:** stochastic/expected monotonic attention boundaries with latency regularization. **Result:** improved quality–latency trade-offs over prior monotonic approaches. **Limitations/relevance:** a gating mechanism can implicitly decide whether evidence is sufficient but does not make semantic alternatives explicit. It motivates a learned “wait/prepare/stroke” gate.

#### X5. Shi et al. — Emformer (ICASSP 2021, published)

**Source:** [paper](https://arxiv.org/abs/2010.10759), [DOI](https://doi.org/10.1109/ICASSP39728.2021.9414560).  
**Problem and inputs:** efficient low-latency ASR with long memory, cached left context, and block processing. **Causality/future:** uses configured real right context and is therefore bounded-lookahead rather than strictly causal. **Representation/horizon/uncertainty:** memory bank plus acoustic blocks; lookahead/average latency is configuration-dependent. **Latency/result:** at 80 ms average latency, reported WER is 3.01/7.09 on LibriSpeech test-clean/test-other; 960 ms settings improve further. **Limitations/relevance:** ASR emission and motion commitment differ. It supplies principled 80 ms and larger actual-lookahead oracle settings.

#### X6. Strimel et al. — ANCAT (ICML 2023, published)

**Source:** [paper](https://arxiv.org/abs/2305.04159).  
**Problem and inputs:** avoid choosing one fixed causal/noncausal streaming-ASR operating point by dynamically using future acoustic context only where beneficial. **Causality/future:** explicitly noncausal; it selects how much **real** future audio to read. **Representation/horizon/uncertainty:** adaptive attention context; no semantic future is generated. **Result:** closes accuracy gaps to noncausal systems while improving latency. **Limitations/relevance:** actual waiting cannot prove predicted lookahead helps. It is an important adaptive-wait upper bound and suggests that gesture forecast use should itself be gated.

#### X7. Narayanan et al. — Cascaded Encoders for Unified Streaming/Non-Streaming ASR (ICASSP 2021, published)

**Source:** [paper](https://arxiv.org/abs/2010.14606), [DOI](https://doi.org/10.1109/ICASSP39728.2021.9414607).  
**Problem and inputs:** share one model between a streaming causal path and a future-aware nonstreaming refinement path. **Causality/future:** first encoder is causal; second consumes the complete first-encoder sequence. **Representation/horizon/uncertainty:** shared acoustic features; full-utterance future in offline mode; no predicted future. **Result:** streaming WER comparable to a standalone streaming model and 10–27% relative WER improvement offline. **Limitations/relevance:** the offline branch is not distilled into the streaming branch by itself. It is a clean backbone layout for noncausal teacher and causal student sharing lower layers.

#### X8. Ekstedt and Skantze — Voice Activity Projection (Interspeech 2022, published)

**Source:** [paper](https://arxiv.org/abs/2205.09812), [project/code](https://erikekstedt.github.io/VAP/).  
**Problem and inputs:** continuously predict turn-taking events from dyadic stereo audio without manual event labels. **Causality/future:** past/current audio only; joint future activity is predicted. **Representation/horizon/uncertainty:** probability over 256 joint states encoding two speakers’ speech/no-speech in 0.2, 0.4, 0.6, and 0.8 s bins, totaling 2 s. **Result:** strong zero-shot turn-shift/backchannel tasks and later real-time demonstrations. **Limitations/relevance:** voice activity is far simpler than gesture semantics. Its discrete joint future and full posterior are an excellent model for speech-act/prosody/gesture-state forecasts.

#### X9. Du et al. — CosyVoice 2 (2024 preprint)

**Source:** [preprint](https://arxiv.org/abs/2412.10117), [code](https://github.com/QwenAudio/CosyVoice).  
**Problem and inputs:** scalable low-latency streaming TTS with a text–speech language model and chunk-aware causal flow matching. **Causality/future:** speech synthesis can stream relative to arriving text, but TTS deployments often possess text ahead of audio; causality relative to an unfinished language plan depends on the upstream interface. **Representation/horizon/uncertainty:** supervised semantic speech tokens and chunk-level acoustics; ordinary generative sampling. **Result:** reported near-lossless streaming quality, human-parity naturalness, and minimal response latency. **Limitations/relevance:** it does not predict unseen conversational text. It is a practical speech decoder if a shared causal planner is prototyped.

#### X10. Labiausse et al. — Hibiki (ICML 2025, published)

**Source:** [OpenReview](https://openreview.net/forum?id=fgjN8B6xVX), [paper](https://arxiv.org/abs/2502.03382).  
**Problem and inputs:** high-fidelity simultaneous speech-to-speech translation that adapts how long it waits for semantic evidence. **Causality/future:** processes the live source stream and accumulates sufficient prefix; it does not claim to guess unseen source. **Representation/horizon/uncertainty:** a multistream decoder jointly emits target text and audio; word-specific delays are weakly supervised using a translation model’s perplexity. **Result:** state-of-the-art reported quality, speaker fidelity, and naturalness with real-time/on-device compatibility and released inference code. **Limitations/relevance:** adaptive lag, not predicted lookahead. It shows that shared text/audio tokens and learned delay can preserve cross-modal consistency.

#### X11. Leviathan, Kalman, and Matias — Speculative Decoding (ICML 2023, published)

**Source:** [PMLR paper](https://proceedings.mlr.press/v202/leviathan23a.html).  
**Problem and inputs:** accelerate exact autoregressive sampling by drafting several tokens cheaply and verifying them in parallel with the target model. **Causality/future:** no external future; candidate future tokens are predicted. **Representation/horizon/uncertainty:** a length-\(K\) draft block and accept/reject probabilities preserve the target distribution exactly. **Commit/revise:** only a verified prefix is emitted; rejected suffixes were never committed. **Latency/result:** 2–3× acceleration on T5-XXL with identical output distribution. **Limitations/relevance:** visual motion cannot be recalled after rendering, and the real future does not arrive as an exact verifier. The transferable principle is a hard uncommitted/committed boundary and consensus-prefix acceptance.

---

## 7. What gesture timing evidence actually implies

### 7.1 A useful quantitative prior, not a fixed offset

Three findings are especially actionable:

- Donnellan et al.’s naturalistic corpus places representational gesture onset about **370 ms before the lexical affiliate on average**.
- Ter Bekke et al. report that **94% of gesture onsets** precede the lexical affiliate, by **644 ms on average** (SD 648 ms); **60% of stroke onsets** precede it, by **193 ms on average** (SD 616 ms). These numbers use different annotation definitions and corpora from Donnellan et al.; they should not be pooled as if they estimate one universal constant.
- Morrel-Samuels and Krauss show that lower lexical familiarity lengthens gesture–speech asynchrony, consistent with a gesture plan being available while lexical retrieval is still underway.

These results make MIBURI forecast horizons of **0.32 and 0.64 s** especially plausible. A 0.96–1.0 s condition tests longer preparations and complex semantic gestures; 0.16 s tests whether a shallow local forecast suffices; 2.0 s is a useful stress test but likely too uncertain for routine semantic commitment.

### 7.2 Phase, function, and reference matter

A gesture unit is not one indivisible event:

- **Preparation** moves the hands from rest or a previous gesture into position. It can begin well before the associated word and is often the safest place to exploit uncertain forecasts.
- **Stroke** carries the most effort/meaning. Its onset, apex, and hold can align with a lexical affiliate, prosodic accent, or broader discourse unit; it should face the strictest confidence threshold.
- **Hold** can preserve a referent or suspend commitment while speech catches up.
- **Retraction** follows the stroke/hold and can extend beyond the affiliate, merge into the preparation of the next gesture, or be interrupted. It should not be forced to fit an artificial forecast window.

Gesture types also have different information requirements:

- **Beats** can be driven largely by local prosody and are a safe fallback.
- **Deictic gestures** require a stable referent and direction; a wrong early commitment is conspicuous.
- **Iconic/metaphoric gestures** need event semantics, viewpoint, and often a longer preparation.
- **Discourse/regulatory gestures** depend on turn structure and interlocutor state, making VAP-style joint forecasts relevant.

### 7.3 Evaluation consequence

“Gesture leads word by \(x\) ms” cannot be the sole target. A better annotation/evaluation tuple is:

\[
(\text{communicative act},\ \text{gesture type},\ \text{referent},\
t_{\rm prep},\ t_{\rm stroke},\ t_{\rm apex},\ t_{\rm hold},\
t_{\rm retract},\ \text{lexical/conceptual affiliate},\ \text{prosodic accent})
\]

The review finds no established co-speech benchmark that provides all of these at MIBURI scale. A carefully annotated semantic/timing subset of BEAT2 is therefore part of the research contribution, not clerical overhead.

### 7.4 Why strictly speech-reactive models tend toward beats

This is partly an information-timing effect, not simply insufficient model size:

1. **Local prosody is available now.** Energy, rhythm, and acoustic onset strongly support beat timing from a short causal window.
2. **Referential semantics may arrive too late.** The entity, spatial relation, negation, or pragmatic force needed for an iconic, metaphoric, deictic, or regulatory stroke may only become unambiguous at or after the lexical affiliate—after preparation should have begun.
3. **Conditional averaging favors low-commitment motion.** Before semantics disambiguate, several meaning-bearing gestures are plausible. Token likelihood can allocate mass to generic frequent motion or beats, which are compatible with many futures.
4. **Visible actions are costly to reverse.** A conservative causal model is rewarded for not making a conspicuous wrong point or depiction.
5. **Training/evaluation reinforce the shortcut.** BeatAlign and generic realism metrics directly reward rhythmic motion, while semantic correctness and contradiction are weakly supervised or unmeasured.
6. **Dataset imbalance matters.** Beats and nonsemantic motion are frequent; rare referent-specific gestures have a long-tailed mapping from speech to form.

An explicit forecast is promising only if it turns some of this uncertainty into a calibrated, actionable plan. Merely shifting generic motion earlier will not solve the semantic problem.

---

## 8. The information-theoretic question: can a forecast from the same past help?

Let the forecast be a deterministic function of the same causal context:

\[
\hat Z=f_\phi(C_t).
\]

Then \(\hat Z\) adds no external information beyond \(C_t\):

\[
I(G_t;\hat Z\mid C_t)=0.
\]

Equivalently, \(\sigma(C_t,\hat Z)=\sigma(C_t)\). For an unrestricted Bayes-optimal generator with unlimited data, compute, and successful optimization, adding \(\hat Z\) cannot lower the irreducible Bayes risk. A stochastic forecast \(\hat Z\sim q(\cdot\mid C_t)\) with only independent sampling noise also adds no information about the actual future once \(C_t\) is known.

This does **not** make the architecture pointless. It makes the correct scientific claim narrower: explicit prediction may improve a finite model’s learning, compute allocation, uncertainty handling, or multimodal coordination.

### 8.1 Legitimate ways it can help

1. **Inductive bottleneck.** A forecast head forces the network to isolate event semantics, prosodic accents, turn state, or gesture phase rather than burying them in an entangled hidden state.
2. **Future supervision.** CPC, action anticipation, and privileged distillation show that targets from later time steps can shape a better causal representation even though they are absent at inference.
3. **Task decomposition.** Predicting a low-rate plan and then rendering motion can be easier than learning a direct prefix→fine-motion map with one loss and one time scale.
4. **Compute allocation.** A short forecast can be produced at a low rate and reused across several 80 ms MIBURI steps, increasing effective semantic planning depth without delaying every motion token.
5. **Multimodality and abstention.** A posterior over several futures tells the generator which aspects have consensus and which should remain reversible. A direct generator trained only by token likelihood may not expose this decision variable.
6. **Upstream hidden plans.** If the gesture module currently observes only emitted/current Moshi states but Moshi possesses an earlier internal response plan not included in \(C_t\), exposing that plan *does* add information from the gesture head’s perspective. The comparison must define \(C_t\) precisely.
7. **Shared stochastic commitment.** If speech and gesture independently sample from \(p(S\mid C)\) and \(p(G\mid C)\), they can choose incompatible modes. Sampling a common latent plan \(P\) first does not add outside information, but it changes the joint factorization to \(p(P\mid C)p(S\mid P,C)p(G\mid P,C)\) and can enforce consistency.
8. **Optimization regularization.** Auxiliary forecasts can reduce shortcut learning, improve gradient signal at semantic time scales, and make errors diagnosable.

### 8.2 When it will not help—or will hurt

- A matched-capacity direct MIBURI already represents the useful future sufficient statistic.
- The forecast target is dominated by unpredictable content, so mutual information \(I(Z_{t:t+H};C_t)\) is low.
- Point forecasts average incompatible modes or a best-of-\(K\) objective rewards diversity without meaningful probabilities.
- Teacher-forced future labels are clean, but inference forecasts are noisy; the generator learns to over-trust a channel that degrades at test time.
- The generator ignores the forecast, so gains come only from extra parameters or training steps.
- Forecast errors trigger irreversible iconic/deictic strokes, creating contradictions worse than a conservative beat gesture.
- Forecast computation consumes the 80 ms budget or delays first motion.
- The chosen horizon is longer than the conversational plan’s stability under interruption, repair, or barge-in.

### 8.3 The indispensable causal ablation

At minimum compare four capacity-matched models:

1. **Direct:** \(G_t\leftarrow C_t\).
2. **Future-supervised hidden state:** \(G_t\leftarrow C_t\), plus an auxiliary future loss whose prediction is not fed back.
3. **Explicit forecast-conditioned:** \(G_t\leftarrow(C_t,\hat Z)\).
4. **Shuffled/corrupted forecast:** same architecture as (3), but forecasts are mismatched or noised at inference.

If (2) matches (3), the benefit is representation learning, not inference-time foresight. If (3) beats (2) and degrades smoothly under calibrated corruption, the exposed forecast is doing decision-relevant work.

---

## 9. Transfer map

| Research family and original-domain problem | Representative papers | Underlying shared mechanism | How it maps to MIBURI | What must change for gesture generation | Critical risk/test |
|---|---|---|---|---|---|
| **Simultaneous translation:** emit target before the source sentence ends | [Gu 2017](https://aclanthology.org/E17-1099/), [Alinejad 2018](https://aclanthology.org/D18-1337/), [wait-k](https://aclanthology.org/P19-1289/), [MMA](https://openreview.net/forum?id=Hyg96gBKPS) | Choose READ/WAIT, WRITE, or PREDICT from a source prefix; discard unverified predicted state | Choose wait/hold, reversible preparation, beat, or semantic stroke; optionally predict future plan tokens | Gestures are continuous and already rendered motion cannot be retracted like an uncommitted token; define physical commit boundaries | Compare to equal-delay waiting; count irreversible wrong strokes |
| **Streaming ASR/TTS:** improve quality with limited latency | [Emformer](https://arxiv.org/abs/2010.10759), [ANCAT](https://arxiv.org/abs/2305.04159), [Cascaded Encoders](https://arxiv.org/abs/2010.14606), [CosyVoice 2](https://arxiv.org/abs/2412.10117) | Fixed/adaptive right context, shared streaming/offline backbones, chunk-aware causal decoding | Construct real-lookahead oracles, adaptive-wait baselines, and a practical shared speech branch | Keep “causal relative to audio,” “causal relative to received text,” and “causal relative to communicative intent” separate | Hidden buffering or preavailable text can masquerade as causal quality |
| **Action anticipation:** recognize an action before it unfolds | [Vondrick 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Vondrick_Anticipating_Visual_Representations_CVPR_2016_paper.html), [RULSTM](https://openaccess.thecvf.com/content_ICCV_2019/html/Furnari_What_Would_You_Expect_Anticipating_Egocentric_Actions_With_Rolling-Unrolling_LSTMs_ICCV_2019_paper.html), [Gammulle 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gammulle_Predicting_the_Future_A_Jointly_Learnt_Model_for_Action_Anticipation_ICCV_2019_paper.pdf), [FUTR](https://openaccess.thecvf.com/content/CVPR2022/html/Gong_Future_Transformer_for_Long-Term_Action_Anticipation_CVPR_2022_paper.html), [RAFTformer](https://openaccess.thecvf.com/content/CVPR2023/html/Girase_Latency_Matters_Real-Time_Action_Forecasting_Transformer_CVPR_2023_paper.html) | Predict task-relevant future embeddings/events, jointly optimize the present task, and charge compute latency | Predict semantic/prosodic/phase events rather than future waveform; train forecast for gesture utility | Shorten horizons to phase-level subsecond windows and model “no gesture” as a valid event | A predictable future feature may be irrelevant to gesture; test oracle value first |
| **Trajectory and human-motion forecasting:** cover several physically plausible futures | [MultiPath](https://arxiv.org/abs/1910.05449), [DLow](https://arxiv.org/abs/2003.08386), [Trajectron++](https://arxiv.org/abs/2001.03093) | Anchored mixture distributions, correlated diverse samples, optional conditioning on the agent’s own plan | \(K\) gesture-intent/prosody anchors with probabilities, residual timing, and common speech–gesture hypothesis ID | Replace geometric road anchors with communicative act/type/referent anchors; enforce upper-body feasibility and semantic calibration | Diversity without calibrated coverage can increase contradictions |
| **Noncausal-teacher → causal-student distillation:** use complete actions/video during training only | [Tran 2021](https://arxiv.org/abs/1904.04868), [PPKD](https://doi.org/10.1016/j.patcog.2022.108741), [Cascaded Encoders](https://arxiv.org/abs/2010.14606) | Distill selected features/logits; progressively widen privileged-future horizon | Full-context EMAGE/RAG-Gesture/noncausal-MIBURI teacher supervises causal MIBURI | Use shift-tolerant phase/semantic losses and partial features because exact future-conditioned states are unattainable | Teacher may teach impossible early timing; compare to explicit forecast and auxiliary-loss controls |
| **Predictive representation learning:** learn causal state from future targets | [CPC](https://arxiv.org/abs/1807.03748) | Contrastively predict future latents during training; discard prediction head at deployment | Add future semantic/prosodic auxiliary objectives to MIBURI encoder | Prevent easy acoustic nuisance targets from dominating; match parameters and training budget | If this matches explicit conditioning, the forecast itself is unnecessary |
| **World models and MPC:** act now using uncertain imagined rollouts | [PETS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [PlaNet](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) | Sample latent futures, score them, execute only the first action, and replan | Reforecast every 80 ms; expose only a short, safe motion prefix | Define communicative reward/constraint, batch short rollouts, and separate preparation from stroke | Model error compounds and rollout compute can miss the gesture deadline |
| **Speculative/rolling generation:** draft several future tokens/frames and verify or refine them | [Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html), [Rolling Diffusion](https://doi.org/10.1609/aaai.v40i31.39807) | Maintain an uncommitted future, accept a prefix, revise the suffix | Buffer future plan/phase tokens; render only consensus or verified prefix | No exact verifier exists for an unobserved conversational future; use calibrated risk and physical commitment rules | Do not claim exactness; measure artifacts when forecasts flip |
| **Turn-taking projection:** anticipate who will speak and when | [VAP](https://arxiv.org/abs/2205.09812) | Predict a full distribution over compact joint future states at several temporal resolutions | Predict speech-act/turn/gesture-state bins for speaker and partner | Extend binary activity to communicative acts, gesture type, phase, and referent without exploding state count | Factorized states may lose dependencies; monolithic codebooks may be sparse |
| **Shared multimodal planning/joint generation:** keep modalities coherent | [de Ruiter](https://doi.org/10.1017/CBO9780511620850.018), [FastTalker](https://doi.org/10.1016/j.neucom.2025.130074), [Match-TTSG](https://arxiv.org/abs/2310.05181), [Gelina](https://arxiv.org/abs/2510.12834) | Sample or expose one upstream plan/state for both outputs | Causal plan tokens drive Moshi speech and MIBURI gesture; share one hypothesis ID | Plans must precede outputs without using a complete script; add revision/version semantics | A common plan can synchronize both modalities around the same wrong guess |
| **Legible robot motion:** reveal intent before task completion | [Dragan et al. 2013](https://doi.org/10.15607/RSS.2013.IX.024) | Optimize action for early observer inference of the intended goal | Score whether gesture helps a viewer infer the upcoming concept/act | Model speech-conditioned human observers and naturalness constraints, not only geometric goal beliefs | A highly legible motion may become exaggerated or unnatural |

---

## 10. Research gaps

1. **No direct forecast-conditioned causal gesture model.** The literature has zero-lookahead reaction, actual lookahead, full-context semantics, and joint scripted synthesis, but not the requested combination.
2. **No standard causality audit.** “Streaming,” “autoregressive,” “causal convolution,” and “real-time” are routinely conflated. Timestamp provenance and prefix-invariance tests are missing.
3. **No calibrated semantic uncertainty.** Gesture generators sample diverse motion but rarely report whether a semantic or deictic commitment is likely to be correct.
4. **No explicit irreversible-action policy.** Existing systems do not formalize when preparation is safe, when a stroke commits meaning, or how to recover from a wrong plan.
5. **Weak semantics/timing metrics.** FGD, BeatAlign, diversity, and general preference tests cannot establish anticipatory appropriateness.
6. **Point forecasts dominate adjacent transfers.** Multimodal trajectory work is mature, but co-speech gesture has not adopted anchors, posterior calibration, selective prediction, or risk-aware fallback.
7. **Shared plan vs joint decoder is unresolved.** Scripted joint systems show benefits, but there is no live causal comparison between shared hidden features, explicit plan tokens, and merely interleaved outputs.
8. **Future distillation is untested for gesture.** MIBURI supplies a causal student and EMAGE/RAG-Gesture/GestureLSM supply candidate teachers, but the information-gap-aware distillation experiment has not been done.
9. **Forecast horizon is not tied to gesture phase.** A single window size ignores the fact that preparation can safely lead while stroke should wait for stronger evidence.
10. **Interactional revision is under-modeled.** Barge-in, self-repair, partner feedback, and referent changes can invalidate a plan within hundreds of milliseconds.
11. **Latency is not measured end to end.** Model-step time, audio buffering, speech-model latency, rendering, and synchronization are often reported separately.
12. **Dataset support is inadequate.** Large motion datasets lack reliable phase, affiliate, referent, communicative-act, and uncertainty annotations at the needed scale.

---

## 11. Candidate architectures

| Architecture | Input information | Forecast / plan representation | Training losses | Causality guarantee | Uncertainty handling | Expected latency | Main failure modes | Required dataset annotations | Most relevant precedents |
|---|---|---|---|---|---|---|---|---|---|
| **A. Downstream Forecast-MIBURI** | Past gesture + timestamped current/past Moshi speech/text states | \(K\) semantic/prosodic/gesture-phase anchors, probabilities, timing residuals | MIBURI token losses + mixture NLL + semantic/phase/prosody + Brier + realization/consistency | Prefix-only masks/cache; prefix-invariance test; no forecast target enters input | Entropy/agreement gate; abstain/beat/hold fallback | Small extra cached head; design for p95 <80 ms step | Ignored forecast, overconfident deictic stroke, mode collapse | Gesture type, phases, affiliate/concept, referent/direction, prosody, plan revisions | MultiPath, VAP, Gammulle, Alinejad, PETS |
| **B. Noncausal teacher / causal student** | Student: MIBURI prefix only; teacher: full utterance during training | No required inference forecast; selected future-aware logits/features | Base token loss + KL logits + partial hidden + shift-tolerant phase/relational distillation | Teacher removed at inference; student masks identical to MIBURI | Student entropy only; no explicit future posterior | Near base MIBURI | Impossible teacher target, overconfident copied timing, teacher bias | Codec tokens; optional type/phase/affiliate labels for interpretable distillation | Tran, PPKD, Cascaded Encoders, EMAGE/RAG-Gesture teachers |
| **C. Shared causal speech–gesture planner** | Live dialogue prefix before both speech and gesture commitment | Low-rate plan token: act, concepts, prosody, gesture type/referent/phase, version/status | Speech + gesture likelihoods + plan supervision + realization + shared-commitment consistency | Plan is generated from prefix before both outputs; timestamp/version audit | \(K\) plans or posterior; both modalities share sampled hypothesis | Removes serial TTS→gesture inference but adds plan step; measure first-speech delay | Both modalities consistently wrong, ignored plan, leaked script | Transcript semantics, discourse act, prosody, gesture type/phases/referent, revision state | de Ruiter, Kita–Özyürek, FastTalker, Match-TTSG, Gelina |
| **D. Speculative joint rollout** | Same causal prefix as A/C | \(K\) joint speech/plan/gesture rollouts plus verifier scores | Draft likelihood/coverage + verifier ranking/calibration + motion feasibility/consistency | Only prefix inputs; not-yet-rendered branches remain private | Mode probability, verifier margin, branch consensus; abstention | Highest; batch branches and commit 80–240 ms maximum | Deadline misses, verifier overconfidence, branch collapse, visible flip artifacts | Same as C plus corrupted/counterfactual rollout labels | PETS, PlaNet, speculative decoding, rolling diffusion |

### Architecture A — Forecast-MIBURI: downstream multi-hypothesis forecast conditioning

**Purpose:** the smallest change that directly tests the hypothesis while preserving Moshi and MIBURI’s causal interface.

#### Inputs and outputs

At MIBURI step \(t\), the forecaster receives only:

- cached Moshi semantic/acoustic hidden states with timestamps \(\le t\);
- past regional gesture RVQ embeddings \(g_{<t}\);
- optional causal dialogue state already available to Moshi (speaker/turn state, not future transcript).

It predicts \(K\) hypotheses \(F_t^k\) and weights \(\pi_t^k\) for horizons \(H\in\{2,4,8,12\}\) MIBURI steps:

\[
F_t^k =
(\text{speech act},\text{event/entity relation},\text{gesture type},
\text{referent/direction},\text{phase-time bins},
\text{pitch/energy/onset envelope}) .
\]

The recommendation is **not** to predict raw waveform or exact future motion first. A hybrid plan has:

- discrete anchors: none/beat/deictic/iconic/metaphoric/regulatory plus communicative act;
- continuous residuals: prosody envelope and time-to-preparation/stroke/apex;
- a “none/abstain” mode;
- probabilities \(\pi^k\) and aleatoric scale for continuous variables.

The current MIBURI temporal transformer cross-attends to candidate embeddings and their horizon positions. A confidence gate produces

\[
\tilde h_t=h_t+\alpha_t\sum_k\pi_t^k\,W F_t^k,
\qquad
\alpha_t=\mathrm{gate}(H(C_t),{\cal H}(\pi_t),\text{agreement}_t).
\]

#### Training

1. **Create future targets.** Derive prosody from future audio; align future words/concepts; annotate or weakly label gesture event/type/phase. All future information is a target only.
2. **Forecaster warm-up.** Train from prefix-only inputs with:

\[
\mathcal L_F =
\lambda_{\rm cls}\mathcal L_{\rm anchor}
+\lambda_{\rm time}\mathcal L_{\rm phase}
+\lambda_{\rm pros}\mathcal L_{\rm prosody\text{-}NLL}
+\lambda_{\rm mix}\mathcal L_{\rm mixture\text{-}NLL}
+\lambda_{\rm cal}\mathcal L_{\rm Brier}.
\]

3. **Generator warm-up with corruption.** Condition on oracle future plans, then progressively replace them with sampled predictions. Randomly drop, shuffle within a controlled window, or corrupt plan components so the generator learns a conservative fallback.
4. **End-to-end fine-tuning.** Add MIBURI’s RVQ token losses and expressiveness objectives:

\[
\mathcal L=\mathcal L_{\rm MIBURI}+\mathcal L_F+
\lambda_{\rm real}\mathcal L_{\rm forecast\rightarrow realization}
+\lambda_{\rm cons}\mathcal L_{\rm speech\text{-}gesture\ consistency}.
\]

`forecast→realization` compares the old forecast with the speech/gesture that actually arrives; `consistency` penalizes deictic/type/entity contradictions. Gradients from the gesture loss may enter the forecast head, but an auxiliary target loss must remain so the forecast stays interpretable.

#### Inference and revision

- Reforecast every 80 ms or every 160 ms with caching.
- Reuse a low-rate plan across intervening MIBURI steps.
- Immediately down-weight a hypothesis when new Moshi evidence lowers its likelihood.
- Never revise rendered frames. Revise only hidden plans, future phase schedules, or not-yet-emitted RVQ tokens.
- If confidence is low, choose neutral/beat motion, continue a hold, or begin only a reversible preparation.
- Lock a semantic hypothesis once its stroke begins; if later evidence conflicts, finish gracefully, retract, and log a repair rather than snapping to a different gesture.

#### Uncertainty and commitment policy

Use both posterior entropy and cross-hypothesis agreement:

- **Low confidence:** no semantic stroke; beats/idle/hold allowed.
- **Moderate confidence with consensus geometry:** initiate preparation toward a shared neutral manifold.
- **High confidence and stable for \(m\) updates:** commit stroke/type/referent.
- **Distribution shift:** forecast dropout fallback to base MIBURI.

Thresholds should be selected on validation selective-risk curves, not tuned by visual preference alone.

#### Latency and implementation

- Implement the forecaster as a small cached transformer/SSM on top of already computed Moshi and MIBURI states.
- Produce all horizons and \(K\) modes in one forward pass, as in MultiPath/FUTR.
- Design target: keep end-to-end p95 below the 80 ms gesture-step deadline; report added p50/p95 time and memory separately from Moshi/rendering.
- A first implementation can add a forecast module and fusion projection without modifying the three codecs or 2D token order.

#### Expected failures

Posterior collapse to beats, candidate modes differing only cosmetically, overconfident deictic errors, generator ignoring the forecast, forecast/generator co-adaptation that destroys calibration, and semantic target noise. Each is directly measurable.

---

### Architecture B — Full-context teacher, causal MIBURI student

**Purpose:** test whether future-aware training helps without an inference-time forecast or latency cost.

#### Teacher and student

- **Teacher:** an offline full-context model such as EMAGE, RAG-Gesture, GestureLSM, or a noncausal copy of MIBURI whose cross-attention sees the complete utterance.
- **Student:** unchanged causal MIBURI input interface and autoregressive decoder.
- **Shared space:** regional codec tokens, temporal hidden states, gesture event/type logits, and phase-time predictions.

#### Losses

\[
\mathcal L_S =
\mathcal L_{\rm MIBURI}
+\lambda_{\rm logit}T^2\,\mathrm{KL}(p_T^T\|p_S^T)
+\lambda_{\rm hid}\sum_{\ell\in{\cal S}}\|P_\ell h_{S,\ell}-\mathrm{sg}(h_{T,\ell})\|_1
+\lambda_{\rm phase}\mathcal L_{\rm phaseKD}
+\lambda_{\rm rel}\mathcal L_{\rm relational}.
\]

Use only selected student features for hidden distillation, following PPKD; the student cannot exactly imitate future-conditioned activations. Train a curriculum of teachers seeing +160, +320, +640, +960 ms, then full context. Shift-tolerant or contrastive alignment is preferable to framewise equality because a causal student may reasonably prepare earlier or later.

#### Inference, uncertainty, and latency

The teacher and future labels disappear. Inference is exactly causal, with only a small optional phase/type head. Student entropy can gate semantic motion, but there is no explicit forecast posterior. Runtime should remain near base MIBURI.

#### Expected value and failure modes

This may recover semantic features at no inference cost and is the strongest alternative explanation for any gain from Architecture A. It cannot conjure unpredictable content. A teacher can also transfer full-context timing habits that are impossible online, causing overconfident early motion; future-corruption and teacher-horizon curricula are safeguards.

---

### Architecture C — Shared causal speech–gesture planner

**Purpose:** make speech and gesture consequences of the same prospective decision, aligning with de Ruiter/Kita–Özyürek and avoiding incompatible independent samples.

#### Factorization and representation

\[
P_j\sim p_\psi(P_j\mid C_{\le j}),\qquad
S_{j:j+H_s}\sim p_{\theta_s}(\cdot\mid C_{\le j},P_j),\qquad
G_{j:j+H_g}\sim p_{\theta_g}(\cdot\mid C_{\le j},P_j).
\]

`P_j` is a low-rate discrete/continuous plan packet generated *before* the corresponding speech and stroke:

- discourse act and intended addressee;
- planned lexical concepts/entities/relations, not necessarily surface words;
- prosodic prominence/duration envelope;
- gesture type, referent, handedness/direction, and phase schedule;
- revision/commit status and hypothesis probability.

Two implementation routes are plausible:

1. add a plan-token head to Moshi and expose it to both Moshi’s speech decoder and MIBURI;
2. interleave `PLAN`, speech, and gesture tokens in one backbone, borrowing Gelina’s token schedule while keeping live-input masks.

#### Training

Weakly supervise plans from transcript semantics, prosody, and gesture annotations; then jointly optimize speech token likelihood, gesture RVQ likelihood, plan prediction, and cross-modal realization:

\[
\mathcal L =
\mathcal L_{\rm speech}
+\mathcal L_{\rm gesture}
+\mathcal L_{\rm plan}
+\lambda_{\rm agree}\mathcal L_{\rm plan\text{-}realization}
+\lambda_{\rm MI}\mathcal L_{\rm shared\ commitment}.
\]

Scheduled plan corruption is necessary. To show that explicit planning matters, compare with a shared-backbone model of equal size that exposes no plan token.

#### Inference and consistency

Sample one plan hypothesis ID and condition both modalities on it. Plans can be revised until either the speech realization or gesture stroke crosses its commitment boundary. Once one modality commits, the other must retain the same entity/relation/type or issue an explicit repair. A versioned finite-state plan record—`tentative → preparing → committed → realized/repaired`—is more reliable than unconstrained hidden-state overwriting.

#### Latency and implementation

Plan packets should run at a lower rate than audio codes and can be predicted one or several 80 ms gesture steps ahead. This architecture removes a serial “generate speech then infer gesture” dependency, but it changes Moshi training and data plumbing, so it is a second-stage project. Measure whether plan emission itself delays first speech.

#### Failure modes

Degenerate plans that merely copy the next word, plans ignored by one decoder, synchronized but uniformly wrong outputs, plan-token exposure leaking future text during training, and cascade failures where one mistaken plan corrupts both speech and motion.

---

### Architecture D — Multi-hypothesis speculative speech–gesture rollout

**Purpose:** test the maximum value of explicit imagined futures and receding-horizon selection.

#### Mechanism

From \(C_t\), a cheap draft model proposes \(K\) short joint rollouts:

\[
\{(\hat S_{t:t+H}^k,\hat P_{t:t+H}^k,\hat G_{t:t+H}^k),\pi_k\}_{k=1}^K.
\]

A verifier scores semantic coherence, gesture feasibility, base-MIBURI likelihood, forecast calibration, and disagreement with the partner/scene. Only a consensus or high-scoring prefix of length \(c \ll H\) can influence visible motion. Replan after every new Moshi block.

#### Training and inference

Train the draft with multi-hypothesis NLL/best-of-many plus coverage; train the verifier on correct, corrupted, and counterfactual rollouts. During inference, batch all branches, cache the common prefix, and commit at most 80–240 ms of reversible motion. Unlike exact speculative decoding, real future speech is not an immediate verifier, so “acceptance” is risk-based rather than distribution preserving.

#### Uncertainty, latency, and failures

Use mode probability, verifier margin, and cross-branch agreement. If branches disagree on gesture type/referent, do not commit a semantic stroke. This architecture has the highest compute and calibration burden; branch collapse, verifier overconfidence, and missed deadlines are likely. It should follow—not precede—the cheaper Architecture A experiment.

---

## 12. Cross-modal consistency and recovery rules

Regardless of architecture:

1. **Share the hypothesis ID.** Do not independently take the most likely speech future and the most likely gesture future if those modes may be incompatible.
2. **Separate reversible and irreversible phases.** A neutral preparation manifold and holds provide time; the stroke/referent is the semantic commitment.
3. **Use versioned plans.** Every output token records the plan version that caused it. New evidence can supersede only uncommitted versions.
4. **Train on forecast errors.** Feed plausible wrong, stale, low-confidence, and out-of-distribution forecasts during training.
5. **Learn a fallback.** Base MIBURI behavior should remain available when the forecast channel is dropped or uncertain.
6. **Measure contradiction explicitly.** Examples include pointing left while saying “right,” depicting increase while speech says decrease, or performing a turn-yield gesture while continuing the turn.
7. **Use realization feedback.** Compare a plan to subsequently realized speech; update calibration by horizon, gesture type, speaker, and discourse context.
8. **Handle interruption.** On barge-in or self-repair, freeze/hold or retract rather than finish a semantically invalid stroke.

---

## 13. Experiment sequence

### Stage 0 — Establish an auditable causal baseline

Before adding a model:

- reproduce MIBURI quality and latency with fixed seeds and documented hardware;
- log a timestamp for every input token, hidden state, forecast, gesture token, decoded frame, and rendered frame;
- test **prefix invariance**: two examples with identical prefixes and different futures must produce identical deterministic states/outputs through the end of the common prefix;
- inspect convolution padding, normalization statistics, data collation, bidirectional encoders, transcript alignment, and cached attention masks;
- distinguish model compute, buffering delay, Moshi delay, codec delay, and rendering delay.

**Gate:** no causal claim proceeds until the prefix-invariance and timestamp tests pass.

### Stage 1 — Measure the value of real future information

Add nondeployable oracle conditions with ground-truth future speech/text at +0.16, +0.32, +0.64, +0.96, and +2.0 s:

- future Moshi/text embeddings;
- future prosody only;
- future semantics only;
- both semantics and prosody.

Also include an equal-delay system that simply waits and then runs a bounded-lookahead model. This estimates:

1. whether future information can improve the relevant gesture measures at all;
2. which component and horizon matter;
3. the maximum forecast-conditioned headroom;
4. whether waiting is cheaper than predicting.

**Gate:** if oracle future does not improve semantic appropriateness/phase timing without unacceptable responsiveness loss, explicit forecasting is not justified.

### Stage 2 — Predictability and target-selection probes

Freeze the causal encoder and train lightweight heads for:

- future acoustic tokens or embeddings;
- pitch/energy/onset envelope;
- next semantic concept/entity relation;
- gesture event/type and phase;
- VAP-style turn-state bins.

Evaluate accuracy, NLL/CRPS, calibration, and coverage versus horizon. Compare raw future targets to compact event/plan targets at matched parameter count.

**Gate:** choose only targets that remain predictably above a strong frequency/last-value baseline and have downstream oracle value. A target can be predictable but useless, or useful but unpredictable.

### Stage 3 — Test the central forecast-conditioning hypothesis

Train, with equal encoder/generator capacity and compute:

| Condition | Future-supervised? | Forecast fed to generator? | Real future at inference? |
|---|---:|---:|---:|
| Base MIBURI | No | No | No |
| Larger MIBURI | No | No | No |
| Auxiliary future loss | Yes | No | No |
| Explicit point forecast | Yes | Yes | No |
| \(K\)-hypothesis forecast | Yes | Yes | No |
| Shuffled forecast | Yes | Yes, mismatched | No |
| Oracle forecast | N/A | Yes | **Yes, diagnostic only** |

Use at least three training seeds and report paired bootstrap confidence intervals over clips/speakers. Primary claims should compare the explicit forecast to both base and auxiliary-loss models, not only to base MIBURI.

### Stage 4 — Uncertainty, corruption, and commitment

Calibrate on held-out speakers. Sweep confidence thresholds and report selective risk/coverage. Inject:

- token substitution and prosody noise;
- hypothesis permutation;
- delayed/stale forecasts;
- simulated ASR/semantic errors;
- mid-utterance referent changes, self-repairs, and interruptions;
- forecast-channel dropout.

Compare always-use, entropy-gated, agreement-gated, and wait/prepare/stroke policies. Record wrong semantic commitments per minute and recovery time.

**Gate:** predicted lookahead should fail gracefully toward base-MIBURI performance, not catastrophically below it.

### Stage 5 — Distilled future awareness

Train Architecture B using teacher horizons from +0.16 s to full context. Compare:

- output-logit distillation only;
- all-layer feature matching;
- selected-layer/partial-feature PPKD;
- progressive-horizon curriculum;
- shift-tolerant phase/semantic distillation.

This decides whether explicit inference-time forecasts justify their latency and complexity.

### Stage 6 — Shared planner

Only after a useful forecast representation is identified:

- train a plan-conditioned speech branch and gesture branch;
- compare independent sampling, shared hypothesis ID, shared hidden features without explicit plan, and explicit plan tokens;
- intervene on a plan token while holding history fixed and verify that both modalities change consistently;
- test plan revisions before and after speech/gesture commitment.

### Stage 7 — Speculative rollout

Batch \(K\) branches, vary rollout \(H\) and committed prefix \(c\), and plot quality/calibration against p95 latency and GPU memory. This stage is justified only if one-step/multi-horizon forecasts are already useful and calibrated.

---

## 14. Evaluation protocol and metrics

### 14.1 Forecast quality

- **Discrete semantics/gesture state:** NLL, top-\(k\) accuracy/recall, macro-F1, Brier score, expected calibration error, adaptive calibration error.
- **Continuous prosody/timing:** Gaussian/mixture NLL, CRPS, MAE for time-to-preparation/stroke/apex, pitch/energy correlation.
- **Multimodal coverage:** oracle-of-\(K\) accuracy, minimum set distance, mode coverage, pairwise diversity, probability mass on the realized mode.
- **Selective prediction:** risk–coverage and area under risk–coverage curves; wrong-commit rate at fixed coverage.
- **Temporal usefulness:** score a correct prediction by when it became available, subtracting compute/buffer time, as motivated by RAFTformer.

### 14.2 Motion realism and stability

Retain MIBURI’s FGD, BeatAlign, L1 diversity, and facial MSE for continuity with prior work, but add:

- velocity, acceleration, jerk, joint-limit violations, foot skating, and root drift;
- boundary discontinuity at forecast/plan revisions;
- frequency and duration of frozen poses, unnecessary movement, and abrupt retractions;
- within-speaker and across-seed diversity.

These are quality controls, not evidence of semantic anticipation.

### 14.3 Semantic and cross-modal correctness

- gesture-type macro-F1 and event detection average precision;
- referent/direction accuracy for deictic gestures;
- semantic retrieval/ranking of the matched speech concept from generated motion;
- human-coded contradiction rate by relation (left/right, up/down, size, count, negation, turn yield/hold);
- plan→speech and plan→gesture realization accuracy;
- mutual consistency when the same hypothesis ID is sampled;
- mismatched-speech discrimination: raters should prefer the matched speech–gesture pair to a plausible mismatched pair.

### 14.4 Phase and timing

- preparation, stroke, apex, hold, and retraction onset/offset MAE;
- generated vs human distribution of gesture-onset and stroke-onset lead relative to lexical/conceptual affiliate;
- Wasserstein distance or distributional tests, rather than only mean offset;
- prosodic-accent/apex alignment and beat synchronization;
- proportion of semantic strokes that begin before sufficient evidence;
- recovery time and trajectory smoothness after a forecast revision.

Report results separately by beat, deictic, iconic/metaphoric, and discourse/regulatory gesture.

### 14.5 Interaction and latency

- p50/p90/p95/p99 model-step time;
- deadline-miss rate relative to MIBURI’s 80 ms step;
- first-gesture and first-semantic-gesture latency;
- end-to-end audio-in→rendered-motion delay;
- forecast age when consumed;
- GPU memory, energy/throughput where feasible;
- barge-in stop latency, plan-cancellation latency, and semantic recovery time;
- quality–latency Pareto front against actual-lookahead/wait baselines.

### 14.6 Human evaluation

Use separate questions, not one overloaded “quality” score:

1. motion naturalness;
2. speech–gesture rhythmic alignment;
3. semantic appropriateness;
4. whether the gesture helps anticipate/understand upcoming content;
5. responsiveness/latency;
6. perceived contradiction or hesitation.

Include:

- matched vs mismatched speech;
- causal base vs forecast model at equal rendered latency;
- forecast model vs bounded real-lookahead oracle;
- silent-motion and audio-only controls where meaningful;
- full clips plus targeted windows around annotated affiliates;
- blinded, randomized paired comparisons and mixed-effects analysis with participant and item effects.

A primary success criterion should require a semantic/timing or human-appropriateness gain **without** a material increase in contradiction, deadline misses, or naturalness loss.

---

## 15. Novelty assessment and likely contribution

### Direct novelty

The defensible claim is:

> A causal co-speech gesture generator that explicitly forecasts a calibrated, multimodal near-future semantic/prosodic plan from the available speech–dialogue prefix, uses that forecast to control current gesture phase and commitment, and revises uncommitted plans as real evidence arrives.

This is distinct from:

- MIBURI/LiveGesture: causal but no exposed forecast;
- STARGATE/rolling gesture diffusion: real future audio lookahead;
- semantic gesture systems: full-text access;
- teacher distillation: future available only in training, with no explicit inference plan;
- scripted joint speech–gesture models: complete text is already known.

### Mechanism novelty

Do **not** claim that “prediction improves a simultaneous system” is new. Alinejad et al. demonstrated the core predict/use/discard mechanism in 2018; action anticipation predicts task-relevant features; PETS/PlaNet replan imagined futures; MultiPath and VAP expose multimodal distributions; PPKD distills privileged future; speculative decoding drafts and verifies future tokens.

### Strongest possible contribution

The most valuable paper would not merely add a forecast module. It would establish:

1. a causality audit for gesture models;
2. phase-aware semantic timing annotations/metrics;
3. oracle future value and prefix predictability by horizon;
4. a matched-capacity proof that exposed forecasts help beyond auxiliary future supervision;
5. calibrated commit/abstain behavior under forecast error;
6. an open MIBURI extension with end-to-end latency and interruption tests.

That package would convert an intuitive idea into a falsifiable account of when prospective information helps embodied generation.

---

## 16. Recommended implementation order

1. Add timestamp logging and prefix-invariance tests to the existing MIBURI path.
2. Build oracle future-semantic and future-prosody adapters at 0.16/0.32/0.64/0.96/2.0 s.
3. Annotate a balanced BEAT2 subset for gesture type, phases, affiliate, referent, and contradiction.
4. Train prefix-only forecast probes; choose compact predictable targets.
5. Implement Architecture A with \(K\) anchors and a forecast-dropout fallback.
6. Run base/larger/auxiliary/explicit/shuffled/oracle controls.
7. Add confidence gating and interruption/corruption tests.
8. Train Architecture B as the zero-inference-cost alternative.
9. Attempt Architecture C only if an explicit plan variable shows measurable utility.
10. Treat speculative rollouts as an optional efficiency/quality extension, not the first proof.

---

## 17. Five papers to read first

1. [MIBURI: Towards Expressive Interactive Gesture Synthesis](https://arxiv.org/abs/2603.03282) — the exact causal baseline, design constraints, and current evaluation gap.
2. [Prediction Improves Simultaneous Neural Machine Translation](https://aclanthology.org/D18-1337/) — the closest precedent for predicting unseen input, acting on it, and resetting when reality arrives.
3. [Hand Gestures Have Predictive Potential During Conversation](https://doi.org/10.1111/cogs.13407) — the most directly relevant phase/timing evidence and a warning that temporal lead alone does not prove benefit.
4. [Progressive Privileged Knowledge Distillation for Online Action Detection](https://doi.org/10.1016/j.patcog.2022.108741) — the strongest alternative to inference-time forecasting and the right response to the teacher/student information gap.
5. [MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses](https://arxiv.org/abs/1910.05449) — a compact, efficient blueprint for multimodal forecasts with explicit mode probabilities and per-time uncertainty.

---

## 18. One-sentence recommendation

Start with a **0.32–0.96 s, multi-hypothesis semantic/prosodic forecast head plus phase-aware confidence gating on unmodified MIBURI**, and make the auxiliary-future-loss and equal-delay real-lookahead baselines strong enough that a positive result really means prospective structure—not extra capacity or hidden latency—improved causal gesture generation.
