# Detecting stereotypes in the GPT-2 language model using CrowS-Pairs

*GPT-2* is a language model that can score how likely it is that some text is a valid English sentence: not only grammaticality, but also the 'meaning' of the sentence is part of this score. *CrowS-Pairs* is a dataset with pairs of more and less stereotypical examples for different social groups (e.g., gender and nationality stereotypes).

Below, you can select a CrowS-Pairs bias type from the drop-down menu, and click `Sample` to sample 10 random pairs from CrowS-Pairs. Alternatively, type your own pair of sentences. The demo shows for each pair of sentences which one receives the higher score ('is more likely').

If a language model systematically prefers more stereotypical examples, this is taken as evidence that has learnt these stereotypes from the training data and shows undesirable bias.
