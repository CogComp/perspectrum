from nltk import sent_tokenize, word_tokenize
import re
from webapp.models import Evidence

# Cleaning evidences (citations, references)

if __name__ == '__main__':

    q = Evidence.objects.all()

    for evi in q:
        text = evi.content
        sents = sent_tokenize(text)
        print(len(sents))

        start_whitelist = {'"', '\''}
        # Rule 0, if sentences start with 'number', it is likely a reference line, drop it
        sents = [s.strip() for s in sents]
        sents = [s for s in sents if s[0].isalpha() or s[0] in start_whitelist]

        # Rule 1, if sentences start with '[', it is likely a reference line, drop it
        sents = [s.strip() for s in sents]
        sents = [s for s in sents if not s.startswith("[")]

        # Rule 2, remove everything enclosed in "[[ ]]"
        sents = [re.sub('\[\[.*\]\]', '', s).strip() for s in sents]

        # Rule 3, remove everything enclosed in "[ ]"
        sents = [re.sub('\[.+\]', '', s).strip() for s in sents]

        # Rule 4, remove sentences starting with enclosed in "[ ]"
        sents = [s for s in sents if not 'http://' in s]

        # Rule 5, remove double spaces
        sents = [s.replace("  ", " ").strip() for s in sents]

        # Rule 6, remove sentences with 5 or less tokens
        sents = [s for s in sents if len(word_tokenize(s)) > 5]

        ft = " ".join(sents)

        evi.content = ft
        evi.save()
