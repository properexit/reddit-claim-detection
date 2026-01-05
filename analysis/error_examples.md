# Qualitative Error Analysis

This section presents representative error cases illustrating linguistic and discourse phenomena that challenge automated claim detection in self-medication discussions.

## Example 1

**Text excerpt**

> Even in U.S where you need a script even for... Hydroxyzine ? Where I live (in EU), in my country all meds (except narcotics and antibiotics) are OTC. If we want to get a med cheap AF, we go to a public doctor (public servant) you pay nothing for the visit, they e-prescribe and later at the pharmacy, you give them the printed prescription and pay only 25% if insured, and completely nothing if uninsured. Otherwise, we go to a pharmacy, ask for the med and mgs we want, pay it 100% its price and vo...

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.598`

- Predicted span: _Even in U.S where you need a script even for... Hydroxyzine ?_


**Linguistic category**

- `narrative_framing`


**Explanation**

The text is primarily descriptive and explanatory, presenting a narrative comparison of healthcare systems rather than advancing a specific propositional claim.


---

## Example 2

**Text excerpt**

> I don't remember the exact cocktail but I remember it was a bunch of Prozac, quanfacine, amitriptyline, and some OTC things. They had just filled their scripts and ODed and then came into the ED with second thoughts. It wasn't their first attempt but they were successful this time.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.597`

- Predicted span: _I don't remember the exact cocktail but_


**Linguistic category**

- `experiential_report`


**Explanation**

The content recounts a past experience without asserting a generalizable or evaluative claim; meaning is anchored in personal observation.


---

## Example 3

**Text excerpt**

> Wow this sounds like a female version of me. Early 20s I started with OTC then Rx codeine for back pain, have had a script ever since but it's never been enough, and OTC codeine has been gone since 2018 here. I'm guessing you're in the UK because of DHC and online scripts, I've been there and when codeine linctus was OTC and before online pharmacies cracked down I'd buy heaps of them. Then I started getting DHC (DF118) from somewhere that had 100x bottles OTC until Glaxo stopped manufacturing th...

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.637`

- Predicted span: _Wow this sounds like a female version of me._


**Linguistic category**

- `narrative_framing`


**Explanation**

This is an extended personal narrative expressing affect and experience rather than making a discrete factual or evaluative claim.


---

## Example 4

**Text excerpt**

> My apologies, I failed to describe the medication change scenario. I started taking Allegra a year ago every today for around three months, there was no decrease in lymph node size and my doctor said it was okay to quit taking it. When I mentioned lymph nodes again the only answer I got was "some people's are just like that." But it's so bothersome that I can't really believe this is a normal occurrence with no underlying cause. At least, not without any imaging or other tests.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.913`

- Predicted span: _I started taking Allegra a year ago every today for around three months, there was no decrease in lymph node size and my doctor said it was okay to quit taking it._


**Linguistic category**

- `implicit_causality`


**Explanation**

The text implies a causal relationship between medication use and symptom persistence without explicitly asserting causation.


---

## Example 5

**Text excerpt**

> We've been cleaning it with soap and water but nothing else until a few hours ago. Started an OTC topic antibiotic cream and wrapping it. He's leaked through 2 wraps in the last few hours. Definitely yellow.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.721`

- Predicted span: _We've been cleaning it with soap and water but nothing else until a few hours ago._


**Linguistic category**

- `experiential_report`


**Explanation**

The utterance reports ongoing actions and observations, leaving interpretation to the reader rather than stating a claim.


---

## Example 6

**Text excerpt**

> Consumming a powerful drug 8 days in a row isn't medicating, to self-medicate you have to have discipline and a good understanding of the effects. Judging by his post this guy is going in the wrong direction for that.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.764`

- Predicted span: _Consumming a powerful drug 8 days in a row isn't medicating, to self-medicate you have to have discipline and a good understanding of the effects._


**Linguistic category**

- `contrast_only`


**Explanation**

The statement relies on definitional contrast and normative framing rather than advancing an empirical or testable claim.


---

## Example 7

**Text excerpt**

> Going by your definition of a hard drug, every prescription drug and over the counter drug could qualify as one. I'm not defending drugs like cocaine, but I feel like your definition is too broad.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.502`

- Predicted span: _Going by your definition of a hard drug, every prescription drug and over the counter drug could qualify as one. I'm not defending drugs like cocaine, but I feel like your definition is too broad._


**Linguistic category**

- `contrast_only`


**Explanation**

The argument is rhetorical and contrastive, critiquing a definition rather than asserting a concrete claim.


---

## Example 8

**Text excerpt**

> Id definitely drink more water and try over the counter migraine medicine and if that still don't help id go to the doctor and talk about getting on anxiety meds .

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.525`

- Predicted span: _Id definitely drink more water and try over the counter migraine medicine and if that still don't help id go to the doctor and talk about getting on anxiety meds ._


**Linguistic category**

- `advice_vs_claim`


**Explanation**

The sentence functions as advice or suggestion rather than a declarative claim.


---

## Example 9

**Text excerpt**

> Otc meds getting you high? Yea ok lmao

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.183`


**Linguistic category**

- `contrast_only`


**Explanation**

The utterance is sarcastic and dismissive, using contrast for rhetorical effect rather than stating a claim.


---

## Example 10

**Text excerpt**

> Most of the mid levels working urgent cares—and probably a few doctors as well—don't even interpret rapid strep tests correctly. A large number of people are chronically colonized with strep. You can swab a completely asymptomatic person and it will come back positive. You can swab someone with sore throat who is having clearly viral symptoms (rhinorrhea, cough, congestion), and they're positive. An OTC test will just be a ticket to unnecessary antibiotic prescribing.

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.960`

- Predicted span: _Most of the mid levels working urgent cares—and_


**Linguistic category**

- `implicit_causality`


**Explanation**

The passage suggests downstream consequences of testing practices without explicitly stating causal claims.


---

## Example 11

**Text excerpt**

> Might be different experience but that's a kids drug. No grown person does dxm… we go out and do drugs that actually feel good not shit you get over the counter at cvs lol

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.602`

- Predicted span: _No grown person does dxm… we go out and do drugs that actually feel good not shit you get over the counter at cvs lol_


**Linguistic category**

- `contrast_only`


**Explanation**

The statement is opinionated and contrastive, expressing attitude rather than a factual or evaluative claim.


---

## Example 12

**Text excerpt**

> You can get this otc easy don't need a scrip but the raspberry flavor messses it up mixing with anything tastes like shi most otc cough syrup with codeine will have the favoring but most part will definitely get you some what taken just str8 like that

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.839`

- Predicted span: _You can get this otc easy don't need a scrip but the raspberry flavor messses it up mixing with anything tastes like shi most otc cough syrup with codeine will have the favoring but most part will definitely get you some what taken just str8 like that_


**Linguistic category**

- `experiential_report`


**Explanation**

The meaning is grounded in personal experience and informal description rather than a structured claim.


---

## Example 13

**Text excerpt**

> Idk but being addicted to brutally overdosing otc medicine doesn't sound fun to me lol

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.635`

- Predicted span: _Idk but being addicted to brutally overdosing otc medicine doesn't sound fun to me lol_


**Linguistic category**

- `opinion_only`


**Explanation**

This is a subjective opinion expressing personal stance rather than an objective or evaluative claim.


---

## Example 14

**Text excerpt**

> Sorry for my ignorance, I'm 66 and about to be accessed then hopefully medicated. By swimming do you mean taking your stimulant medication? I have a long history of self medication with amphetamines and stimming sounds like getting on it.

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.331`


**Linguistic category**

- `question_or_clarification`


**Explanation**

The text primarily consists of clarification questions rather than declarative statements.


---

## Example 15

**Text excerpt**

> Most of the warnings on over the counter drugs are so if some idiot takes like 10,000mg at once the drug companies are protected from lawsuits.

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.429`


**Linguistic category**

- `implicit_causality`


**Explanation**

The sentence implies legal and corporate motivations without explicitly stating a causal claim.


---

## Example 16

**Text excerpt**

> I was doing the same as you, part prescription, part OTC… I was self medicating…

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.481`


**Linguistic category**

- `experiential_report`


**Explanation**

This is a brief experiential alignment with another user rather than a claim.


---

## Example 17

**Text excerpt**

> Do not take antibiotics without a prescription. What do you mean by OTC antibiotics?

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.138`


**Linguistic category**

- `question_or_directive`


**Explanation**

The utterance mixes a directive with a clarification question, not a factual claim.


---

## Example 18

**Text excerpt**

> Ibuprofen~ Advil or Motrin are a couple brand names. You can buy ibuprofen generic. Or Aleve or Naproxen OTC.

**Model behavior**

- Gold label: `1`

- Predicted label: `0`

- Confidence: `0.460`


**Linguistic category**

- `factual_statement`


**Explanation**

The sentence lists factual information without argumentative or evaluative intent.


---

## Example 19

**Text excerpt**

> Hi OP. The symptoms you describe are classic for a simple UTI... you will need antibiotics...

**Model behavior**

- Gold label: `0`

- Predicted label: `1`

- Confidence: `0.625`

- Predicted span: _Hi OP. The symptoms you describe are classic for a simple UTI... you will need antibiotics..._


**Linguistic category**

- `implicit_causality`


**Explanation**

The diagnosis is implied based on symptoms, suggesting causality without explicit medical justification.


---
