# Total number of Extracted Facts: 194
# True: 163/194=0.84
# Major False: 23/194=0.12
# Minor False: 8/194=0.04


# We select DeepSeek-R1 and OpenAI o1 as representatives of the current SOTA LLMs at the time of writing this article. Due to the absence of fine-grained, long-form hallucination datasets for these two models, we construct two new datasets: OpenAI-o1-WikiBioFacts and DeepSeek-R1-WikiBioFacts.
# These datasets comprise biographies generated for 21 randomly selected notable figures from WikiBio, all of whom have Wikipedia pages but lack comprehensive online biographies.
# Using greedy decoding, we prompt each model to generate a biography for the corresponding figure and employ GPT-4o to automatically extract factual claims.
# OpenAI o1 produces 168 claims, while DeepSeek-R1 generates 194.
# Each claim is manually verified and labeled as true or false.
# Statistical analysis reveals 41 false claims in OpenAI-o1-WikiBioFacts and 31 in DeepSeek-R1-WikiBioFacts.
# We apply hallucination detection methods to identify erroneous claims, though some inaccuracies may not be confabulations.

# After our article is accepted, we will release the complete dataset. If you need the complete dataset before then, please contact me via email at xtaozhao@163.com.
