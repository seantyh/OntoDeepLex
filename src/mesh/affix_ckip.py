from itertools import chain
from lxml import etree

class Affix:
    def __init__(self, elem):
        self._affix_type = "affix"
        for child in elem.iterchildren():
            tag = child.tag.lower()
            value = child.text
            if value:
                value = value.strip()

                if tag == "example":
                    value = [x[:-1].split("(") for x \
                            in value.split(",") if len(x)>1]            
            setattr(self, tag, value)
    
    @property
    def affix_type(self):
        return self._affix_type

class Prefix(Affix):
    def __init__(self, elem):
        super().__init__(elem)
        self._affix_type = "prefix"

    def __repr__(self):
        return f"<Prefix: {getattr(self, 'affix')}>"
    
    @property
    def prefix(self):
        return getattr(self, 'affix')

class Suffix(Affix):
    def __init__(self, elem):
        super().__init__(elem)
        self._affix_type = "suffix"

    def __repr__(self):
        return f"<Suffix: {getattr(self, 'suffix')}>"

    @property
    def affix(self):
        return getattr(self, 'suffix')

class CkipAffixes:
    def __init__(self, affix_dir):
        self.base_dir = affix_dir
        prepend_path = lambda x: affix_dir / x
        prefix_files = ["詞首1.xml", "詞首2.xml"]
        suffix_files = ["詞尾1.xml", "詞尾2.xml"]
        load_affix = self.load_affix
        self.prefixes = list(chain.from_iterable(
                        map(lambda x: load_affix(prepend_path(x)), 
                        prefix_files)))
        self.suffixes = list(chain.from_iterable(
                        map(lambda x: load_affix(prepend_path(x)), 
                        suffix_files)))

    def load_affix(self, fpath):
        with fpath.open("r", encoding="UTF-8") as fin:
            root = etree.parse(fin).getroot()

        data = []
        for elem in root.xpath("//affix"):            
            data.append(Prefix(elem))
        
        for elem in root.xpath("//suffix"):
            data.append(Suffix(elem))
        
        return data
    
    def query(self, word):
        prefix_iter = filter(lambda x: x.prefix==word, self.prefixes)
        suffix_iter = filter(lambda x: x.suffix==word, self.suffixes)
        return list(chain(prefix_iter, suffix_iter))



