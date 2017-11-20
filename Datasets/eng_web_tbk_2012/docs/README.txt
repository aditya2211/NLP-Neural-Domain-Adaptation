English Web Treebank
CatalogID: LDC2012T13
Release date: August 1, 2012
Linguistic Data Consortium
Authors: Ann Bies, Justin Mott, Colin Warner, Seth Kulick

1.0 Introduction

This release of the English Web Treebank consists of 254,830
word-level tokens (16,624 sentence-level tokens) of web text in 1174
files annotated for sentence- and word-level tokenization,
part-of-speech, and syntactic structure.  The data is roughly evenly
divided across five genres: weblogs, newsgroups, email, reviews, and
answers.

This data is consistent with the sentence-level tokenization outlined
in WebtextTBAnnotationGuidelines.pdf, with the word-level tokenization
developed cross-site for English Treebanks under the DARPA GALE
project, and with the annotation guidelines for the English
Translation Treebanks at
http://projects.ldc.upenn.edu/gale/task_specifications/EnglishXBank/.
Additional guidelines developed specifically to account for novel
constructions, etc. in the web text genre can be found in
docs/WebtextTBAnnotationGuidelines.pdf.

Google Inc. and the Linguistic Data Consortium collaborated on the
creation of this resource, funded through a gift from Google.


2. Annotation

2.1 Tasks and Guidelines

This release includes tokenization, part-of-speech, and syntactic
treebank annotation.

The data is consistent with the sentence-level tokenization outlined
in WebtextTBAnnotationGuidelines.pdf, with the word-level tokenization
developed cross-site for English Treebanks under the DARPA GALE
project, and with the annotation guidelines for the English
Translation Treebanks at
http://projects.ldc.upenn.edu/gale/task_specifications/EnglishXBank/.
Additional guidelines developed specifically to account for novel
constructions, etc. in the web text genre can be found in
docs/WebtextTBAnnotationGuidelines.pdf.

"Sentence" level tokens are separated by line breaks, and introduced
by <en=#> delimiters.

"Word" level tokens are separated by white space.

All brackets except for "(" and ")" are converted out of
-LCB-/etc. form, and all html character codes have been converted to
their corresponding characters.

Bracket representation is as follows:

() are represented as -LRB- and -RRB- in the .tree files (this
includes emoticons like :--RRB- )

[] {} <> and all other brackets are unchanged.

In addition, we have converted out of the .html character codes at all
levels of annotation:

&quot;        "
&gt;        >
%lt;         <
&amp;        &

A note on the sentence-level tokenization:

The source data for previous English Treebanks created at LDC has been
the output of of either careful human editing or translation.  As
such, line breaks in the source data provided a reliable starting
point for syntactic annotation.  Because of this, an almost entirely
automated system was used to determine the boundaries between
sentences for those corpora.  However, these deterministic methods of
splitting sentences are not wholly adequate for web text data.  We
have therefore developed new guidelines for sentence-level
tokenization for the English Web Treebank, which are included in
WebtextTBAnnotationGuidelines.pdf.

The resulting files in the
data/<GENRE>/source/source_text_ascii_tokenized directories are the data
that feeds into the rest of the treebank annotation pipeline.

2.2 Annotation Process

Text data was extracted from the original source files by script,
taking the linguistic text only from the title and the body, or the
equivalent, in each genre (i.e., not including other headers, markup,
metadata, etc.).

Higher level ASCII characters or non-ASCII characters in the text were
reduced to comparable common ASCII characters, as detailed in
docs/edit.list.

Tokenization by script was manually corrected to be consistent with
tokenization guidelines developed for English Treebanks in the GALE
project and with WebtextTBAnnotationGuidelines.pdf.

The tokenized data was run through an automatic POS tagger, and the
tagger output was manually corrected to be consistent with the English
Treebank part-of-speech annotation guidelines in
http://projects.ldc.upenn.edu/gale/task_specifications/EnglishXBank/
and WebtextTBAnnotationGuidelines.pdf.

The corrected POS annotated data was run through an automatic parser,
and the parser output was manually corrected to be consistent with the
English Treebank syntactic annotation guidelines in
http://projects.ldc.upenn.edu/gale/task_specifications/EnglishXBank/
and WebtextTBAnnotationGuidelines.pdf.

The first QC process consists of a series of specific searches for
approximately 200 types of potential inconsistency and parser or
annotation error.  Any errors found in these searches were hand
corrected.

Lead annotators for this project were Justin Mott and Colin Warner.
The additional annotator was John Laury.

With this project we have added an additional QC process that
identifies repeated text and structures, and flags non-matching
annotations.  Annotation errors found in this way have been manually
corrected.  This error detection system is work in progress and still
in the process of refinement, but we are pleased to be able to include
it in the annotation pipeline for the first time with this project.
The following papers describe this process:

Seth Kulick, Ann Bies, and Justin Mott
Using Derivation Trees for Treebank Error Detection
ACL 2011, Portland, Oregon, USA, June 19-24, 2011
http://papers.ldc.upenn.edu/ACL2011/DerivationTrees_TBErrorDetection.pdf

Seth Kulick and Ann Bies
A TAG-derived Database for Treebank Search and Parser Analysis
TAG+10: 10th International Workshop on Tree Adjoining Grammars and Related Formalisms, New Haven, CT, June 10-12, 2010
http://papers.ldc.upenn.edu/TAG2010/tag-paper-correct.pdf


3. Source Data Profile

3.1 Data Selection Process

3.1.1 Weblog

The source data for Weblog was weblog data previously collected for
other projects by LDC.

Data selection criteria for Weblog were as follows.  Texts were
selected to be

1. In the weblog domain
2. Taken from existing LDC data collections
3. Free of technical and formatting difficulties, to the extent possible
4. Appropriate and on target content (i.e., not spam, not adult content)
5. The entire text of the post, excluding quoted material from previous posts
6. Linguistic text only (i.e., text from HEADLINE and POST fields, or the
   equivalent, not POSTER, POSTDATE, etc.)

3.1.2 Newsgroups

The source data for Newsgroups was newsgroup data previously collected
for other projects by LDC.

Data selection criteria for Newsgroups were as follows.  Texts were
selected to be

1. In the newsgroup domain
2. Taken from existing LDC data collections
3. Free of technical and formatting difficulties, to the extent possible
4. Appropriate and on target content (i.e., not spam, not adult content)
5. The entire text of the post, excluding quoted material from previous posts
6. Linguistic text only (i.e., text from HEADLINE and POST fields, or the
   equivalent, not POSTER, POSTDATE, etc.)

3.1.3 Email

The source data for this subcorpus was selected from the EnronSent
Corpus, a public domain corpus of email text prepared by Will Styler
at the University of Colorado at Boudler:

Styler, Will (2011). The EnronSent Corpus. Technical Report 01-2011,
University of Colorado at Boulder Institute of Cognitive Science,
Boulder, CO.

A description of the EnronSent Corpus from the website
http://verbs.colorado.edu/enronsent/ is as follows:

"The EnronSent corpus is a special preparation of a portion of the
Enron Email Dataset designed specifically for use in Corpus
Linguistics and language analysis. Divided across 45 plain text files,
this corpus contains 2,205,910 lines and 13,810,266 words."

LDC hand selected the data to be annotated for our project's subcorpus
using the following criteria:

1. Approximately 1000 words from each data file in the Colorado set
(starting at randomly generated line numbers in each file, and
manually taking c. 1000 words of appropriate text starting with the
first message that begins after that line number, and going forward),
which yields a bit over 40K total for annotation.

2. We used subject line and body text only (i.e., did not include
headers, etc., parallel to our WB and NG subcorpora data selection).

3. We avoided quoted text, forwarded text and spam when possible
(i.e., hand selected the text/messages that were not quoted, etc.,
also as with the WB and NG data selection), but some repeated text
remains in this subcorpus.  For this reason, we increased the total
amount of data selected for this subcorpus, compared to the other web
text subcorpora.

All of this data was already converted to ASCII as part of the
University of Colorado at Boudler public domain release.  In other web
text subcorpora, we have reduced higher level ASCII characters or
non-ASCII characters in the text to comparable common ASCII
characters, as detailed in docs/edit.list.  However, this step was not
necessary for the Email subcorpus.

3.1.4 Reviews

The source data for Reviews was supplied by Google from internal
reviews collections.

Data selection criteria for Reviews were as follows.

1. Filtered for being sourced by Google's internal review collection
service.  These are local service reviews (e.g., hotels, restaurants,
etc).

2. Filtered for English. This is based on what the user enters or what
Google predicts based on the users locality.  Individual reviews that
were primarily non-English were removed from the annotated portion of
the corpus.

3. Filtered to remove SSNs, credit cards, phone numbers and IP addresses.

4. Only contains the title and body of the review (no dates, user
names, service category, merchant name, etc.).

5. Sampled from the full set of 2.5M reviews randomly ensuring that
each review at least has some body text (the reviews that were empty,
i.e., just a star rating, were thrown out).

6. Each review was assigned a sequential number (0-399999).

7. Reviews totaling approximately 50K words were selected for treebank
annotation.  This resulted in a total of 723 files (one review per
file).  A few were hand-selected, but more than 90% were randomly
selected.

3.1.5 Answers

The source data for Answers was collected by LDC for this project.
Using Yahoo's API, we harvested Questions, which include both the
Subject field and the Content field, and Answers, which include both
the Chosen Answer (if an answer was marked as "chosen" on Yahoo!
Answers, which is not always the case), and two additional Answers (or
up to three total, if three answers were present).

These fields are marked with the following metadata in the originally
harvested files, the .xml files in
/data/answers/source/source_original.

<Question>
<Subject>
</Subject>
<Content>
</Content>
</Question>
<ChosenAnswer>
</ChosenAnswer>
<Answers>
<Answer>
</Answer>
</Answers>

This metadata is not included or annotated in the treebanked files.

Data selection criteria for the annotated data are as follows:

Questions and answers were selected for annotation.  The question
categories were chosen both by the annotators and randomly.
Question/answer pairs within the categories were chosen randomly, and
then further selected manually as follows.

1. Relevance -- answers should be relevant to the questions.

2. English -- questions and answers entirely in a language other than
English were eliminated from the corpus.  Questions and answers with
brief excursions into non-English were kept in the corpus if it was
judged to be expected and acceptable foreign language usage within
English.

3. Answers present -- questions in the corpus have at least one answer
present, with the following three exceptions.  These three files are
included in the corpus, although they include only questions and not
answers:

20111106112223AAmR2im_ans.xml
20111106193025AA2h62V_ans.xml
20111107173321AASNJjc_ans.xml

In all remaining files, up to three total answers were harvested, if
that many answers were available.

4. Appropriate content -- questions and answers in the corpus were
manually filtered for appropriate content.


3.2 Data Sources and Epochs

3.2.1 Weblog

The data consists of English weblog text from various sources dating
from 2003 to 2006.

3.2.2 Newsgroups

The data consists of English newsgroup text from various sources
dating from 2003 to 2006.

3.2.3 Email

The data consists of English email text from the EnronSent Corpus,
dating approximately from the late 1990's and early 2000's.

3.2.4 Reviews

The data consists of English Google review text.  (Dates may be
unknown.)

3.2.5 Answers

The data consists of English Yahoo! Answers text.  (Dates are
unknown.)


4. Annotated Data Profile

The data consists of 1174 files of web text, and a total of 254,830
word-level tokens.


5. Data Directory Structure

A listing of all of the files in this release can be found in
docs/file.tbl.  A listing of the data filenames can be found in
docs/file.ids.<GENRE>.

See section 6. File Format Description below for details on the
various file formats.

The data directory structure is as follows:

./data/<GENRE>/source/source_original -- the text from the original 
       source data or the original source files as collected
./data/<GENRE>/source/source_text_ascii_tokenized -- linguistic text 
       only, extracted from the subject and body fields (or the 
       equivalent), manually annotated with sentence and word level 
       tokenization, with any higher-level ASCII characters or 
       non-ASCII characters changed into common ASCII characters 
       (details in docs/edit.list)
./data/<GENRE>/xml -- TreeEditor xml stand-off annotation files, 
       referencing the source_text_ascii_tokenized .txt files 
       by character offset
./data/<GENRE>/penntree -- the annotation files in Penn Treebank 
       bracketed list style


6. File Format Description

6.1 *.txt (in data/<GENRE>/source/source_text_ascii_tokenized/)

Tokenized source files.  These contain one sentence per line, with all
token boundaries represented by whitespace.  Additionally, each line
starts with a sentence ID number of the form <en=1>.  This is a
formatting requirement for our current version of the TreeEditor
annotation tool.

Sample sentence from
blogspot.com_aggressivevoicedaily_20060629164800_ENG_20060629_164800.

<en=10> I 'll post highlights from the opinion and dissents when I 'm finished .

6.2 *.xml (in data/<GENRE>/xml/)

TreeEditor .xml stand-off annotation files.  These files contain only
the POS and Treebank annotation and reference the .txt files by
character offset.  TreeEditor typically references the original source
file rather than tokenization output as we are doing here.  We made
this switch for these corpora to aid in the handling of sentences
spanning multiple lines in the source text.

6.3 *.tree (in data/<GENRE>/penntree/)

Bracketed tree files following the basic form (NODE (TAG token)).  Each
sentence is surrounded by a pair of empty parentheses.  Sample:

( (S (NP-SBJ (PRP I)) (VP (MD 'll) (VP (VB post) (NP (NP (NNS highlights)) (PP (IN from) (NP (DT the) (NN opinion) (CC and) (NNS dissents)))) (SBAR-TMP (WHADVP-9 (WRB when)) (S (NP-SBJ (PRP I)) (VP (VBP 'm) (ADJP-PRD (JJ finished)) (ADVP-TMP-9 (-NONE- *T*))))))) (. .)) )


7. Data Validation

Care is taken to maintain the integrity of the data at each step.


8. DTDs

The DTD files for the AG are kept in the same directory where the xml
files are: data/<GENRE>/xml/


9. Copyright Information

Portions (c) 2011, 2012 Trustees of the University of Pennsylvania


10. Contact Information

Contact info for key project personnel: 
Ann Bies, Senior Research Coordinator, Linguistic Data Consortium, 
bies@ldc.upenn.edu


11. Update Log

This index was updated on August 1, 2012 by Ann Bies.
