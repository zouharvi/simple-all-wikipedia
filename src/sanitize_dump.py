#!/usr/bin/env python3

import argparse
import pickle
import re
import tqdm
import html
import nltk

args = argparse.ArgumentParser()
args.add_argument(
    "-d", "--data",
    default="data/cswiki-20221101-pages-articles-multistream.xml"
)
args.add_argument(
    "-o", "--output",
    default="data/cswiki-20221101-pages-articles-multistream.pkl"
)
args.add_argument(
    "-cl", "--count-lines",
    action="store_true",
    help="Make an extra pass to get the total number of lines in the file."
)
args = args.parse_args()

IS_ALPHANUMERIC = re.compile('[0-9a-zA-Z]')
FIND_ID = re.compile('.*<id>([0-9]+)</id>.*')
REPLACE_BRACKETS = re.compile(r'\[\[.+?\|(.+?)\]\]')
REPLACE_LINKS = re.compile(r'\[\[(.+?)\]\]')
REPLACE_CURLY = re.compile(r'\{\{([^\|:]+?)\}\}')
REPLACE_CURLY_FULL = re.compile(r'\{\{.+?\}\}')
REPLACE_ANCHOR_LINKS = re.compile(r'&lt;(.+)&gt;')
REPLACE_BRACKET_LINKS = re.compile(r'\[[^\s]+ (.+?)\]')

# sent = "u. Abstrahováním některých z&nbsp;těchto strukturálních vlastností vznikly pojmy okruh, těleso a další. Studiem těchto abstraktních konceptů se zabývá vektorových prostorů, jež v&nbsp;sobě kombinují tři ze čtyř okruhů zájmu matematiky – kvantitu, strukturu a prostor. Diferenciální a integrální počet přidává k&nbsp;těmto třem okruhům i&nbsp;čtvrtý – změnu."
# print(html.unescape(sent))
# exit()

def is_line_textual(line):
    first_char = line[0]
    if IS_ALPHANUMERIC.match(first_char):
        return True 
    else:
        return False

def is_line_id(line):
    return "<id>" in line

def sanitize_stack(stack):
    if len(stack) == 0:
        return
    stack = [REPLACE_BRACKETS.sub(r"\1", x) for x in stack]
    stack = [REPLACE_LINKS.sub(r"\1", x) for x in stack]
    stack = [REPLACE_CURLY.sub(r"\1", x) for x in stack]
    stack = [REPLACE_CURLY_FULL.sub(r"", x) for x in stack]
    stack = [REPLACE_ANCHOR_LINKS.sub("", x) for x in stack]
    stack = [REPLACE_BRACKET_LINKS.sub(r"\1", x) for x in stack]
    stack = [html.unescape(html.unescape(x)) for x in stack]
    stack = [x for x in stack if "{" not in x and "}" not in x and "|" not in x]
    stack = nltk.sent_tokenize(" ".join(stack))
    return stack

if args.count_lines:
    print("Geting number of lines")
    with open(args.data, "r") as f:
        line_count=len([0 for _ in f])
else:
    line_count = None

data = {}

print("Processing")
with open(args.data, "r") as f:
    stack = []
    cur_id = None
    for line in tqdm.tqdm(f, total=line_count):
        if cur_id is None and is_line_id(line):
            cur_id = FIND_ID.match(line).group(1)
        if is_line_textual(line):
            stack.append(line.rstrip("\n"))
        if "</siteinfo>" in line:
            stack = []
            cur_id = None
            continue
        if "</page>" in line:
            stack = sanitize_stack(stack)
            if stack is not None and len(stack) > 0:
                data[cur_id] = stack
            # TODO: this can be paralelized
            # input()
            stack = []
            cur_id = None

print("Found", len(data), "articles")
print("Found", sum([len(x) for x in data.values()]), "lines")

with open(args.output, "wb") as f:
    pickle.dump(data, f)
