import dataclasses
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from functools import cache
from pathlib import Path


@dataclasses.dataclass
class Accumulator:
    add_numer: int = 0
    p_add_denom: int = 0
    r_add_denom: int = 0
    keep_numer: float = 0.0
    p_keep_denom: int = 0
    r_keep_denom: float = 0.0
    del_numer: float = 0.0
    p_del_denom: int = 0
    r_del_denom: float = 0.0

    def __add__(self, other):
        return type(self)(
            self.add_numer + other.add_numer,
            self.p_add_denom + other.p_add_denom,
            self.r_add_denom + other.r_add_denom,
            self.keep_numer + other.keep_numer,
            self.p_keep_denom + other.p_keep_denom,
            self.r_keep_denom + other.r_keep_denom,
            self.del_numer + other.del_numer,
            self.p_del_denom + other.p_del_denom,
            self.r_del_denom + other.r_del_denom)

    @classmethod
    @cache
    def from_count(cls, i, r_avg, o):
        ch_o = (o > 0)
        ch_i_bar = (i == 0)
        max_ch_r = (r_avg > 0) # same to "any(r > 0 for r in rs)"
        return cls(
            int(ch_o and ch_i_bar and max_ch_r),
            int(ch_o and ch_i_bar),
            int(ch_i_bar and max_ch_r),
            min(i, r_avg, o),
            min(i, o),
            min(i, r_avg),
            max(0, i - max(o, r_avg)),
            max(0, i - o),
            max(0, i - r_avg))


@dataclasses.dataclass
class NgramStat:
    in_dict: dict
    ref_dict_list: list[dict]
    out_dict: dict

    def __getitem__(self, key):
        i = self.in_dict.get(key, 0)
        rs = [x.get(key, 0) for x in self.ref_dict_list]
        r_avg = sum(rs) / len(rs)
        o = self.out_dict.get(key, 0)
        return i, r_avg, o

    def keys(self):
        return set.union(
                set(self.in_dict),
                *[set(x) for x in self.ref_dict_list],
                set(self.out_dict))

    def accumlate(self):
        lst = [
            Accumulator.from_count(*self[key])
            for key
            in self.keys()]
        return sum(lst, start = Accumulator())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', dest = 'in_path')
    parser.add_argument('-o', nargs = '+', dest = 'out_path_list')
    parser.add_argument('-r', nargs = '+', dest = 'ref_path_list')
    parser.add_argument('-n', type = int, default = 4)
    parser.add_argument('-t', choices = ['char', 'word'], default = 'word', dest = 'tokenize')
    parser.add_argument('-d', type = int, default = 2, dest = 'digit')
    parser.add_argument('-b', type = float, default = 1.0, dest = 'beta')
    parser.add_argument('-s', action = 'store_true', dest = 'sent')
    parser.add_argument('-v', action = 'store_true', dest = 'verbose')
    return parser.parse_args()


def main(args):
    set_tokenization(args)
    if args.sent:
        s_main(args)
    else:
        c_main(args)


def c_main(args):
    ond_accum = load(args)
    alpha = beta_to_alpha(args.beta)
    for o, nd_accum in enumerate(ond_accum):
        name = Path(args.out_path_list[o]).name
        p_add, r_add, p_keep, r_keep, p_del, r_del, f_add, f_keep, f_del, sari = calc_sari(nd_accum, alpha)
        if args.verbose:
            line = '\t'.join([
                name,
                round_half_up(100 * p_add, args.digit),
                round_half_up(100 * r_add, args.digit),
                round_half_up(100 * p_keep, args.digit),
                round_half_up(100 * r_keep, args.digit),
                round_half_up(100 * p_del, args.digit),
                round_half_up(100 * r_del, args.digit),
                round_half_up(100 * f_add, args.digit),
                round_half_up(100 * f_keep, args.digit),
                round_half_up(100 * f_del, args.digit),
                round_half_up(100 * sari, args.digit)])
            print(line)
        else:
            print(name + '\t' + round_half_up(100 * sari, args.digit))


def s_main(args):
    ond_accum = load(args)
    alpha = beta_to_alpha(args.beta)
    for o, nd_accum in enumerate(ond_accum):
        name = Path(args.out_path_list[o]).name
        dn_accum = list(zip(*nd_accum))
        sari = np.mean([calc_sent_sari(n_accum, alpha)[-1] for n_accum in dn_accum])
        print(name + '\t' + round_half_up(100 * sari, args.digit))


def load(args):
    d_in = load_text(args.in_path)
    dr_ref = list(zip(*[load_text(x) for x in args.ref_path_list]))
    od_out = [load_text(x) for x in args.out_path_list]
    assert len(d_in) == len(dr_ref)
    assert all(len(d_in) == len(d_out) for d_out in od_out)
    return aggreg(args.n, d_in, dr_ref, od_out)


def load_text(path):
    with open(path) as f:
        data = [x.rstrip('\n') for x in f]
    return data


def aggreg(max_n, d_in, dr_ref, od_out):
    return [[[
        make_accum(n, i, r_ref, o)
        for i, r_ref, o in zip(d_in, dr_ref, d_out)]
        for n in range(1, max_n + 1)]
        for d_out in od_out]


@cache
def make_accum(n, i, rs, o):
    stat = NgramStat(
            ngram_counter(n, i),
            [ngram_counter(n, r) for r in rs],
            ngram_counter(n, o))
    return stat.accumlate()


@cache
def ngram_counter(n, sent):
    return dict(Counter(split_ngram(n, sent)))


split_ngram = None


def set_tokenization(args):
    global split_ngram
    assert split_ngram is None # resetting split_ngram is prohibited
    split_ngram = split_char_ngram if args.tokenize == 'char' else split_word_ngram


def split_word_ngram(n, sent):
    sent = sent.split()
    return [' '.join(sent[i : i + n]) for i in range(len(sent) - n + 1)]


def split_char_ngram(n, sent):
    return [sent[i : i + n] for i in range(len(sent) - n + 1)]


def calc_sari(nd_accum, alpha):
    n_accum = [sum(d_accum, start = Accumulator()) for d_accum in nd_accum]
    return calc_sent_sari(n_accum, alpha)


def calc_sent_sari(n_accum, alpha):
    p_add = calc_p_add(n_accum)
    r_add = calc_r_add(n_accum)
    p_keep = calc_p_keep(n_accum)
    r_keep = calc_r_keep(n_accum)
    p_del = calc_p_del(n_accum)
    r_del = calc_r_del(n_accum)
    f_add = calc_f(p_add, r_add, alpha)
    f_keep = calc_f(p_keep, r_keep, alpha)
    f_del = calc_f(p_del, r_del, alpha)
    sari = (f_add + f_keep + f_del) / 3
    return p_add, r_add, p_keep, r_keep, p_del, r_del, f_add, f_keep, f_del, sari


def calc_p_add(n_accum):
    return np.mean([div(x.add_numer, x.p_add_denom) for x in n_accum])


def calc_r_add(n_accum):
    return np.mean([div(x.add_numer, x.r_add_denom) for x in n_accum])


def calc_p_keep(n_accum):
    return np.mean([div(x.keep_numer, x.p_keep_denom) for x in n_accum])


def calc_r_keep(n_accum):
    return np.mean([div(x.keep_numer, x.r_keep_denom) for x in n_accum])


def calc_p_del(n_accum):
    return np.mean([div(x.del_numer, x.p_del_denom) for x in n_accum])


def calc_r_del(n_accum):
    return np.mean([div(x.del_numer, x.r_del_denom) for x in n_accum])


def calc_f(p, r, alpha):
    if p == 0 or r == 0:
        f = 0
    else:
        f = 1 / (alpha / p + (1 - alpha) / r)
    return f


def beta_to_alpha(beta):
    return 1 / (beta ** 2 + 1)


def div(numer, denom):
    return numer / denom if denom > 0 else 0


def round_half_up(x, digit):
    if x == np.inf or x == -np.inf:
        return str(x)
    digit = Decimal('0.' + '0' * digit)
    x = Decimal(str(x)).quantize(digit, rounding = ROUND_HALF_UP)
    return str(x)


if __name__ == '__main__':
    args = parse_args()
    main(args)

