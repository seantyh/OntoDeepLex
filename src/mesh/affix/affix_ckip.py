from itertools import chain, cycle
from lxml import etree

class Affixoid:
    def __init__(self, elem):
        self._affix_type = "affixoid"
        for child in elem.iterchildren():
            tag = child.tag.lower()
            value = child.text
            if value:
                value = value.strip()

                if tag.startswith("example") and tag[-1] != "f":
                    value = self.parse_example(value)

            if tag == "affix":
                self.affixoid = value
                self.position = 0
            elif tag == "suffix":
                self.affixoid = value
                self.position = 1

            setattr(self, tag, value)

    def __hash__(self):
        h_wordnum = hash(getattr(self, "wordnum"))
        h_affixoid = hash(getattr(self, "affixoid"))
        return h_wordnum ^ (h_affixoid << 2)

    def parse_example(self, txt):
        tokens = txt.split(',')
        words_iter = (x.split('(')[0] for x in tokens)
        words = [x for x in words_iter if x]
        return words

class Affix(Affixoid):
    def __init__(self, elem):
        super().__init__(elem)

    def __repr__(self):
        if self.position == 0:
            return f"<Affix(Prefix): {getattr(self, 'affixoid')}>"
        else:
            return f"<Affix(Suffix): {getattr(self, 'affixoid')}>"

    @property
    def affixoid_type(self):
        return ("prefix", "suffix")[self.position]

    @property        
    def example_words(self): 
        example = getattr(self, "example")       
        if not example:
            ex_list = []
        else:
            ex_list = [(self.affixoid_type, ex) for ex in 
                    getattr(self, "example")]
        return ex_list

class Root(Affixoid):
    def __init__(self, elem):
        super().__init__(elem)


    def morpho_rules(self):
        mr_list = []
        for idx in range(1, 6):
            if hasattr(self, f"mr{idx}"):
                mr_list.append((
                    getattr(self, f"mr{idx}"),
                    getattr(self, f"example{idx}")
                    ))
        return mr_list

    def __repr__(self):
        if self.position == 0:
            return f"<Root(Start): {getattr(self, 'affixoid')}>"
        else:
            return f"<Root(End): {getattr(self, 'affixoid')}>"

    @property
    def affixoid_type(self):
        return ("start-root", "end-root")[self.position]

    @property
    def example_words(self):
        ex_list = []
        for idx in range(1, 6):
            if not hasattr(self, f"mr{idx}"):
                continue
            mr = getattr(self, f"mr{idx}")
            examples = getattr(self, f"example{idx}")
            if examples:
                ex_list.extend([(mr_x, ex) for mr_x, ex in 
                    zip(cycle([mr]), examples)])
        return ex_list

class CkipAffixoids:
    def __init__(self, affix_dir):
        self.base_dir = affix_dir
        prepend_path = lambda x: affix_dir / x
        affixoid_files = ["詞首1.xml", "詞首2.xml",
                          "詞尾1.xml", "詞尾2.xml"]
        load_affix = self.load_affix
        self.affixoids = list(chain.from_iterable(
                        map(lambda x: load_affix(prepend_path(x)),
                        affixoid_files)))
        self.affixoid_index = self.index_affixoid()
        self.example_index = self.index_examples()

    def load_affix(self, fpath):
        with fpath.open("r", encoding="UTF-8") as fin:
            root = etree.parse(fin).getroot()

        data = []
        for elem in root.xpath("//affix"):
            data.append(Affix(elem))

        for elem in root.xpath("//suffix"):
            data.append(Affix(elem))

        for elem in root.xpath("//affix2"):
            data.append(Root(elem))

        for elem in root.xpath("//suffix2"):
            data.append(Root(elem))

        return data

    def __len__(self):
        return len(self.affixoids)

    def __iter__(self):
        return iter(self.affixoids)

    def __getitem__(self, idx):
        return self.affixoids[idx]

    def index_affixoid(self):
        index = {}
        for aff in self.affixoids:
            form = aff.affixoid
            index.setdefault(form, []).append(aff)
        return index

    def index_examples(self):
        index = {}
        for aff in self.affixoids:
            for mr, ex in aff.example_words:
                index.setdefault(ex, []).append((mr, aff))
        return index

    def query(self, word, subset=None):
        iter = filter(lambda x: x.affixoid==word, self.affixoids)
        if subset:
            iter = filter(lambda x: x.affixoid_type==subset, iter)
        return list(iter)

    def prefixes(self):
        iter = filter(lambda x: x.affixoid_type=="prefix", self.affixoids)
        return list(iter)

    def suffixes(self):
        iter = filter(lambda x: x.affixoid_type=="suffix", self.affixoids)
        return list(iter)

    def rootsStart(self):
        iter = filter(lambda x: x.affixoid_type=="start-root", self.affixoids)
        return list(iter)

    def rootsEnd(self):
        iter = filter(lambda x: x.affixoid_type=="end-root", self.affixoids)
        return list(iter)




