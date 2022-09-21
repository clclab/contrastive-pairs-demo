# Detecting stereotypes in the GPT-2 language model using CrowS-Pairs

*GPT-2* is a language model which can score how likely it is that some text is a valid English sentence: not only grammaticality, but also the 'meaning' of the sentence is part of this score. *CrowS-Pairs* is a dataset with pairs of more and less stereotypical examples for different social groups (e.g., gender and nationality stereotypes).

You can either select a CrowS-Pairs bias type from the drop-down below and click `Sample`, and then we
sample 10 random pairs from CrowS-Pairs and show whether the stereotypical example gets
a higher score ('is more likely').

**If GPT-2 systematically prefers the stereotypical examples, it has probably learnt these stereotypes from the training data.**
