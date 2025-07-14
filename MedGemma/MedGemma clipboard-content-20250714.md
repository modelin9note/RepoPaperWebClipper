MedGemma Technical Report
Google Research and Google DeepMind 1
Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment
are challenging due to healthcare’s diverse data, complex spectrum of possible tasks, and the
need to preserve privacy. Foundation models that perform well on various medical tasks and require less
task-specific tuning data are critical to accelerating the development of AI for healthcare applications. In
this technical report, we introduce MedGemma, a new collection of medical vision–language foundation
models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and
reasoning on images and text, significantly exceeding the performance of similar-sized generative models
and approaching the performance of task-specific models, while maintaining the general capabilities of
the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvements on
medical multimodal question answering, 15.5-18.1% improvements on chest X-ray finding classification,
and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma
further improves performance in subdomains, reducing errors in electronic health record information
retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods
for pneumothorax classification and histopathology patch type classification. We additionally introduce
MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual
understanding capabilities of MedGemma and, as an encoder, it achieves performance comparable to or
better than specialized medical image encoders. Taken together, the MedGemma collection provides
a strong foundation of medical image and text capabilities, with potential to significantly accelerate
medical research and development of downstream applications. More details about the MedGemma
collection, including tutorials and instructions for downloading the model weights, can be found at
https://goo.gle/medgemma.
1

1. Introduction
The landscape of modern healthcare is characterized by the generation and use of an unprecedented
volume and diversity of data. Diagnosis, treatment, and monitoring rely on synthesizing information
from disparate sources and specialties. Recently developed large multimodal models (LMMs), trained
on massive and diverse datasets, exhibit remarkable capabilities in detecting complex patterns,
generating coherent text, and processing visual information (Achiam et al., 2023; Alayrac et al., 2022;
Chen et al., 2022; Liu et al., 2023, 2024; OpenAI, 2023; Touvron et al., 2023). These capabilities
mark a potential paradigm shift in assisting with current workflows and extracting novel insights.
While general-purpose (non-medically tuned) LMMs demonstrate impressively broad abilities,
generic models can lack nuanced medical understanding and the ability to interpret and reason about
medical data in a robust way (Han et al., 2023; Labrak et al., 2024; Singhal et al., 2023b,c; Toma
et al., 2023; Tu et al., 2024; Yang et al., 2024). Recognizing this gap, we created MedGemma, a
new suite of open, medically-tuned, vision-language foundation models. These models represent the
latest addition to the Health AI Developer Foundations (Kiraly et al., 2024) collection. Built upon the
robust architecture of Gemma 3 (Gemma-Team et al., 2025), the MedGemma models are designed
to interpret and reason about medical images and text while retaining the strong general-purpose
capabilities present in Gemma 3.

1. Introduction
The landscape of modern healthcare is characterized by the generation and use of an unprecedented
volume and diversity of data. Diagnosis, treatment, and monitoring rely on synthesizing information
from disparate sources and specialties. Recently developed large multimodal models (LMMs), trained
on massive and diverse datasets, exhibit remarkable capabilities in detecting complex patterns,
generating coherent text, and processing visual information (Achiam et al., 2023; Alayrac et al., 2022;
Chen et al., 2022; Liu et al., 2023, 2024; OpenAI, 2023; Touvron et al., 2023). These capabilities
mark a potential paradigm shift in assisting with current workflows and extracting novel insights.
While general-purpose (non-medically tuned) LMMs demonstrate impressively broad abilities,
generic models can lack nuanced medical understanding and the ability to interpret and reason about
medical data in a robust way (Han et al., 2023; Labrak et al., 2024; Singhal et al., 2023b,c; Toma
et al., 2023; Tu et al., 2024; Yang et al., 2024). Recognizing this gap, we created MedGemma, a
new suite of open, medically-tuned, vision-language foundation models. These models represent the
latest addition to the Health AI Developer Foundations (Kiraly et al., 2024) collection. Built upon the
robust architecture of Gemma 3 (Gemma-Team et al., 2025), the MedGemma models are designed
to interpret and reason about medical images and text while retaining the strong general-purpose
capabilities present in Gemma 3.

Figure 1 | Overview of the MedGemma model collection featuring the MedSigLIP image encoder, MedGemma
4B Multimodal and MedGemma 27B Text

In this report, we focus on two MedGemma models: a 4B variant that can accept text, images,
or both as input, and a 27B variant that is optimized for text-only inputs. Both models output text.
MedGemma 4B demonstrates strong performance on Vision Question Answering (VQA) benchmarks
compared to prior SOTA models like Med-Gemini (Saab et al., 2024; Yang et al., 2024) despite
being considerably smaller. Both MedGemma 4B and 27B are highly competitive on challenging
text-only medical benchmark tasks, including MedQA (Jin et al., 2021), MedMCQA (Pal et al., 2022),
PubMedQA (Jin et al., 2019), MMLU Med (Hendrycks et al., 2020), AfriMed-QA (Olatunji et al.,
2024), and AgentClinic (Schmidgall et al., 2024) when compared against other open models of
similar scale. In addition to these strong out-of-the-box capabilities, we show how performance can be
further improved by fine-tuning MedGemma on subdomains like chest X-ray reporting, histopathology
classification, and electronic health record information retrieval.

An additional MedGemma variant, a multimodal version of MedGemma 27B, was also developed
and is being released along with the other models. More thorough evaluation of this multimodal 27B
variant is ongoing and preliminary results can be found in Appendix Section F. Unless otherwise
noted in this report, evaluations that reference “MedGemma 27B” refer to the text-only variant of
MedGemma 27B.
In addition to the MedGemma models, we describe the standalone MedSigLIP 400M-parameter
medical image encoder. MedSigLIP is based on SigLIP-400M (Zhai et al., 2023) and is the same
encoder that powers MedGemma’s image interpretation capabilities. When used on its own, MedSigLIP
enables data-efficient and zero-shot image classification and retrieval, with performance comparable
to or exceeding specialized image encoders.
A high level overview of the released models is shown in Fig. 1. More details about the MedGemma
collection, including tutorials and links to download all of the above models, can be found at https:
//goo.gle/medgemma.

2. Methods
2.1. Datasets
For general purpose data replay during pretraining, original data mixtures from SigLIP (Zhai et al.,
2023) and Gemma 3 (Gemma-Team et al., 2025) were leveraged. The medical training and evaluation
datasets largely followed the datasets in Med-Gemini (Yang et al., 2024). In this section, we outline
the specific changes or differences in datasets relative to Med-Gemini.

2.1.1. Training datasets
Text-only datasets: For text datasets, we sampled responses and logits from a large IT (instructiontuned)
teacher using the train splits of multiple medical QA datasets, including MedQA (Jin et al.,
2021), MedMCQA (Pal et al., 2022), PubMedQA (Jin et al., 2019), MedExpQA (Alonso et al., 2024),
AfriMed-QA (Olatunji et al., 2024), HealthSearchQA (Singhal et al., 2023a), and LiveQA (Abacha
et al., 2017). We also sampled responses and logits for approximately 200,000 synthetic medical
questions generated by asking the same large IT teacher to generate a new question using 5 randomly
sampled questions from the above datasets as examples.

Multimodal datasets: Relative to Med-Gemini, the multimodal capabilities of MedGemma are
currently focused on 2D medical images (e.g. X-ray, 2D slices from CT/MRI); 3D volumes and genomic
datasets described in Yang et al. (2024) were not included. Additionally, we and others have identified
potential data quality issues in PathVQA and MedVQA. Thus, we removed them from the training
dataset. We did not include PAD-UFES-20 in the post-training dataset since it focuses on 6-class
classification of very specific lesion types, which is not in line with the goal of more general purpose
dermatology capabilities and use cases. For the PMC-OA component of the training data, we only
included the single panel medical images from PMC-OA for better data quality. Relative to Med-
Gemini we also introduced a larger internal collection for ophthalmology (184,852 more retinal
fundus images), dermatology (51,049 more dermatology images with 210 different skin conditions),
histopathology (a total of ∼32.5 million patch-text pairs), and radiology data (54,573 more CT 2D slices, 47,622 more MRI 2D slices). The additional CT and MRI slices utilized for training were
curated based on mention of a specific slice associated with abnormal findings in the radiology report.

2.1.2. Data Preprocessing
Our data preparation followed Yang et al. (2024) closely. Image padding and resizing algorithms
remain the same, but because the vision encoder is different in Gemma 3, our images were resized
to 896×896 instead of 768×768. Following Gemma 3, we use the SentencePiece tokenizer with
262,000 entries. Additionally, for CT images, we preselected three windows and converted them into
the RGB color channels of the input image to highlight (1) bone and lung, window-width: 2250,
window-level: -100; (2) soft tissue, window-width: 350, window-level: 40; (3) brain, window-width:
80, window-level: 40.

2.2. Modeling Methodology
2.2.1. Modeling Architecture and Training Infrastructure
The MedGemma model architecture follows Gemma 3 (Gemma-Team et al., 2025) and is compatible
with all existing Gemma infrastructure. The vision encoder for Gemma 3 is the 400M variant of
the SigLIP encoder (Zhai et al., 2023) and is shared across the different Gemma language model
sizes (4B, 27B). The input image resolution is 896×896 with pixel values normalized to [-1, 1]. The
language model component also follows Gemma 3, featuring arbitrary image-text interleaving and
long context (128k). Similar to Gemma 3, MedGemma was trained on TPUv4, TPUv5e, and TPUv5p,
leveraged pre-computed visual tokens for memory saving, and used data and model shardings for
multi-pod training.
4