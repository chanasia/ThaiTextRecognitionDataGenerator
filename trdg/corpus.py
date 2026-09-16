"""Build a realistic mixed Thai / English / numeric corpus for text-recognition training.

The stock dictionaries are single words. A CTC recognizer trained on single words falls
apart on real lines (forms, invoices, chat, books) that mix Thai runs, English tokens,
digits, dates, IDs and punctuation. This module composes such lines from:

  * the Thai word list            (trdg/dicts/th.txt)
  * the English word list         (trdg/dicts/en.txt)
  * optional natural text files   (--text, e.g. Thai Wikipedia from tools/fetch_thai_wiki.py)
  * document templates            (form labels, addresses, prices, phone/ID numbers, ...)

Every output line is NFC-normalised, restricted to a charset, length-bounded and cut only
at positions that keep Thai grapheme clusters intact.

Usage:
    python -m trdg.corpus --count 200000 --out out/corpus.txt
    python -m trdg.corpus --count 500000 --text out/thai_wiki.txt my_docs.txt
"""

import argparse
import os
import random
import re
import string
import sys
import unicodedata
from typing import Iterable, Iterator, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- charset
THAI_CONSONANTS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"
THAI_SIGNS = "ะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ๎ฯ"
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ASCII_PRINTABLE = string.digits + string.ascii_letters + string.punctuation + " "
EXTRA = "฿©®°·“”‘’•–—™…"
DEFAULT_CHARSET = frozenset(THAI_CONSONANTS + THAI_SIGNS + THAI_DIGITS + ASCII_PRINTABLE + EXTRA)

# characters that must never start a chunk / end a chunk when cutting
_FOLLOWERS = frozenset("ะัาำิีึืุู็่้๊๋์ํ๎ๅๆฯ")
_LEADING = frozenset("เแโใไ")
_ZERO_WIDTH = re.compile("[​‌‍﻿­]")
_WS = re.compile(r"\s+")
_TO_THAI_DIGITS = str.maketrans("0123456789", THAI_DIGITS)

# --------------------------------------------------------------------------- vocab
TITLES = ["นาย", "นาง", "นางสาว", "ด.ช.", "ด.ญ.", "ดร.", "ผศ.", "รศ.", "ศ.", "คุณ", "ร.ต.อ.", "พ.ต.ท."]
LABELS = [
    "ชื่อ", "ชื่อ-สกุล", "ชื่อบิดา", "ชื่อมารดา", "ที่อยู่", "ที่อยู่ปัจจุบัน", "อีเมล", "Email",
    "เบอร์โทรศัพท์", "โทร", "โทรศัพท์", "มือถือ", "เลขประจำตัวประชาชน", "เลขที่", "วันที่",
    "วันเดือนปีเกิด", "อายุ", "หน่วยงาน", "ตำแหน่ง", "จำนวน", "ราคา", "รวม", "รวมทั้งสิ้น",
    "ยอดสุทธิ", "หมายเหตุ", "เรื่อง", "เรียน", "ลงชื่อ", "ผู้รับ", "ผู้ส่ง", "รหัสสินค้า",
    "รหัสไปรษณีย์", "จังหวัด", "อำเภอ", "ตำบล", "สาขา", "เลขที่บัญชี", "ธนาคาร", "ผู้ขาย",
    "ผู้ซื้อ", "เลขประจำตัวผู้เสียภาษี", "สถานะ", "เลขที่ใบสั่งซื้อ", "วันที่ออกเอกสาร",
    "เงื่อนไขการชำระเงิน", "ผู้ติดต่อ", "Line ID", "รหัสพนักงาน", "แผนก", "สัญชาติ", "ศาสนา",
]
PROVINCES = [
    "กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "ชลบุรี", "ระยอง", "เชียงใหม่",
    "เชียงราย", "ลำปาง", "พิษณุโลก", "นครสวรรค์", "ขอนแก่น", "นครราชสีมา", "อุดรธานี",
    "อุบลราชธานี", "สุรินทร์", "บุรีรัมย์", "สงขลา", "ภูเก็ต", "สุราษฎร์ธานี", "นครศรีธรรมราช",
    "กระบี่", "ตรัง", "ราชบุรี", "กาญจนบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์", "พระนครศรีอยุธยา",
    "สระบุรี", "ลพบุรี", "ฉะเชิงเทรา", "ปราจีนบุรี", "จันทบุรี", "ตราด", "นครปฐม", "สมุทรสาคร",
]
MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม",
          "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
MONTHS_ABBR = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
UNITS = ["บาท", "บาทถ้วน", "ชิ้น", "กล่อง", "ชุด", "คน", "ราย", "ปี", "เดือน", "วัน", "ครั้ง",
         "กก.", "กิโลกรัม", "เมตร", "ตร.ม.", "ไร่", "ลิตร", "น.", "ขวด", "แพ็ค", "หน่วย", "%"]
BANKS = ["ธนาคารกรุงเทพ", "ธนาคารกสิกรไทย", "ธนาคารไทยพาณิชย์", "ธนาคารกรุงไทย",
         "ธนาคารกรุงศรีอยุธยา", "ธนาคารทหารไทยธนชาต", "ธนาคารออมสิน"]
PARTICLES = ["ครับ", "ค่ะ", "นะคะ", "นะครับ", "จ้า", "ด้วยครับ", "ด้วยค่ะ"]
TECH_TOKENS = ["PDF", "Email", "Server", "Docker", "Wi-Fi", "iPhone", "Android", "QR Code",
               "Line", "Facebook", "Excel", "Word", "AI", "OK", "VAT", "ATM", "GPS", "USB",
               "LED", "COVID-19", "HR", "IT", "CEO", "Google", "YouTube", "Windows", "Mac",
               "API", "SMS", "OTP", "e-mail", "Shopee", "Lazada", "Grab", "Netflix", "TikTok"]
DOC_PHRASES = [
    "ขอแสดงความนับถือ", "จึงเรียนมาเพื่อโปรดทราบ", "จึงเรียนมาเพื่อโปรดพิจารณา",
    "ตามที่ท่านได้แจ้งความประสงค์", "ทั้งนี้ ตั้งแต่บัดนี้เป็นต้นไป", "สำเนาถูกต้อง",
    "ใบเสร็จรับเงิน/ใบกำกับภาษี", "รวมเป็นเงินทั้งสิ้น", "ภาษีมูลค่าเพิ่ม 7%", "ยอดรวมก่อนภาษี",
    "ชำระเงินภายในวันที่", "ที่อยู่สำหรับจัดส่ง", "ผู้มีอำนาจลงนาม", "สถานะ: ชำระแล้ว",
    "กรุณากรอกข้อมูลให้ครบถ้วน", "ข้าพเจ้าขอรับรองว่าข้อความข้างต้นเป็นความจริงทุกประการ",
    "ประกาศ ณ วันที่", "ห้ามเข้าก่อนได้รับอนุญาต", "เปิดทำการ จันทร์-ศุกร์ 08.30-16.30 น.",
    "ราคาต่อหน่วย", "จำนวนเงิน (บาท)", "หมายเหตุ: ราคานี้รวมภาษีมูลค่าเพิ่มแล้ว",
    "โปรดเก็บใบเสร็จไว้เป็นหลักฐาน", "แบบคำร้องขอตรวจสอบประวัติ", "ฝ่ายทะเบียนและประมวลผล",
    "เอกสารแนบท้ายสัญญา", "ผู้รับผิดชอบโครงการ", "รายงานการประชุมครั้งที่",
    "ข้อมูลข่าวสารส่วนบุคคล", "บัตรประจำตัวประชาชน", "หนังสือรับรองการทำงาน",
    "ใบรับรองแพทย์", "ทะเบียนบ้านเลขที่", "สำนักงานเขต", "องค์การบริหารส่วนตำบล",
    "กรมสรรพากร", "สำนักงานประกันสังคม", "โรงพยาบาลส่งเสริมสุขภาพตำบล",
]
LIST_MARKERS = ["1.", "2.", "3.", "๑.", "๒.", "๓.", "ก.", "ข.", "ค.", "(1)", "(2)", "ข้อ 1", "ข้อ ๒", "•", "-", "1)", "2)"]


# --------------------------------------------------------------------------- helpers
def normalize(text: str) -> str:
    """NFC + Thai-specific canonical fixes. Apply the SAME function to OCR output when scoring."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ํา", "ำ")          # nikhahit + sara aa  -> sara am
    text = text.replace(" ", " ")
    text = _ZERO_WIDTH.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def safe_cut(text: str, max_len: int) -> str:
    """Cut to <= max_len chars without splitting a Thai grapheme cluster. Prefers a space."""
    if len(text) <= max_len:
        return text
    lo = max(1, max_len // 2)
    cut = text.rfind(" ", lo, max_len + 1)
    if cut > 0:
        return text[:cut].rstrip()
    i = max_len
    while i > lo and (text[i] in _FOLLOWERS or text[i - 1] in _LEADING):
        i -= 1
    return text[:i].rstrip()


def _load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return [l.strip() for l in f if l.strip()]


def load_charset(path: Optional[str]) -> frozenset:
    if not path:
        return DEFAULT_CHARSET
    chars = set()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line:
                chars.add(line if len(line) == 1 else line[0])
    chars.add(" ")
    return frozenset(chars)


# --------------------------------------------------------------------------- builder
class CorpusBuilder:
    def __init__(
        self,
        words: Sequence[str],
        en_words: Sequence[str],
        natural: Sequence[str] = (),
        charset: frozenset = DEFAULT_CHARSET,
        min_len: int = 6,
        max_len: int = 60,
        thai_digit_prob: float = 0.08,
        natural_weight: float = 0.45,
        seed: Optional[int] = None,
    ):
        self.rng = random.Random(seed)
        self.words = [
            normalize(w) for w in words
            if 1 < len(w) <= 20 and set(w) <= charset and w[-1] not in _LEADING
            and (w[0] not in _FOLLOWERS or w.startswith("ฯ"))  # ฯลฯ / ฯพณฯ are real words
        ]
        self.short_words = [w for w in self.words if len(w) <= 8] or self.words
        self.en = [w for w in en_words if w.isalpha() and 2 <= len(w) <= 12] or ["data", "form", "test"]
        self.natural = list(natural)
        self.charset = charset
        self.min_len, self.max_len = min_len, max_len
        self.thai_digit_prob = thai_digit_prob
        self.natural_weight = natural_weight if self.natural else 0.0
        self._prev = None
        self._prev_count = 0

    # ---- primitives
    def _r(self, a, b):
        return self.rng.randint(a, b)

    def _pick(self, seq):
        return self.rng.choice(seq)

    def _p(self, prob):
        return self.rng.random() < prob

    def digits(self, n: int) -> str:
        return "".join(self._pick(string.digits) for _ in range(n))

    def maybe_thai_digits(self, s: str) -> str:
        return s.translate(_TO_THAI_DIGITS) if self._p(self.thai_digit_prob) else s

    def thai_phrase(self, k: Optional[int] = None) -> str:
        k = k or self._r(2, 7)
        out = ""
        for i in range(k):
            w = self._pick(self.words)
            if i and self._p(0.25):        # Thai spaces phrases, not words
                out += " "
            out += w
        return out

    def thai_name(self) -> str:
        first = self._pick(self.short_words)
        last = self._pick(self.words) if self._p(0.5) else self._pick(self.short_words) + self._pick(self.short_words)
        title = self._pick(TITLES) + ("" if self._p(0.7) else " ")
        return f"{title}{first} {last}"

    def english_token(self) -> str:
        if self._p(0.35):
            return self._pick(TECH_TOKENS)
        w = self._pick(self.en)
        return self._pick([w, w.capitalize(), w.upper()])

    def english_line(self) -> str:
        toks = [self.english_token() for _ in range(self._r(2, 7))]
        s = " ".join(toks)
        if self._p(0.3):
            s += self._pick([".", ",", "!", "?", ":"])
        return s

    # ---- numeric things
    def price(self) -> str:
        n = self._pick([self._r(1, 999), self._r(1_000, 99_999), self._r(100_000, 9_999_999)])
        s = f"{n:,}" if self._p(0.6) else str(n)
        if self._p(0.5):
            s += self._pick([".00", ".50", f".{self._r(0, 99):02d}"])
        return self.maybe_thai_digits(s) + self._pick([" บาท", " บาท", "", " ฿", "บาท", " THB", " บาทถ้วน"])

    def phone(self) -> str:
        pats = [
            lambda: "0" + self._pick("689") + self.digits(1) + "-" + self.digits(3) + "-" + self.digits(4),
            lambda: "0" + self._pick("689") + self.digits(8),
            lambda: "0 " + self.digits(4) + " " + self.digits(4),
            lambda: "02-" + self.digits(3) + "-" + self.digits(4),
            lambda: "+66 " + self._pick("689") + self.digits(1) + " " + self.digits(3) + " " + self.digits(4),
        ]
        return self.maybe_thai_digits(self._pick(pats)())

    def id13(self) -> str:
        d = self.digits(13)
        s = f"{d[0]}-{d[1:5]}-{d[5:10]}-{d[10:12]}-{d[12]}" if self._p(0.6) else d
        return self.maybe_thai_digits(s)

    def date(self) -> str:
        d, m = self._r(1, 31), self._r(0, 11)
        y_be, y_ce = self._r(2480, 2575), self._r(1937, 2032)
        pats = [
            lambda: f"{d} {MONTHS[m]} {y_be}",
            lambda: f"{d} {MONTHS[m]} พ.ศ. {y_be}",
            lambda: f"{d} {MONTHS[m]} พ.ศ.{y_be}",
            lambda: f"{d} {MONTHS_ABBR[m]} {y_be % 100:02d}",
            lambda: f"{d} {MONTHS_ABBR[m]} {y_be}",
            lambda: f"วันที่ {d} เดือน {MONTHS[m]} พ.ศ. {y_be}",
            lambda: f"วัน{self._pick(DAYS)}ที่ {d} {MONTHS[m]} {y_be}",
            lambda: f"{d:02d}/{m + 1:02d}/{y_be}",
            lambda: f"{d:02d}/{m + 1:02d}/{y_ce}",
            lambda: f"{d:02d}-{m + 1:02d}-{y_ce}",
            lambda: f"{y_ce}-{m + 1:02d}-{d:02d}",
            lambda: f"{d} {MONTHS_EN[m]} {y_ce}",
            lambda: f"{MONTHS_EN[m]} {d}, {y_ce}",
        ]
        return self.maybe_thai_digits(self._pick(pats)())

    def time(self) -> str:
        h, mi = self._r(0, 23), self._r(0, 59)
        return self.maybe_thai_digits(self._pick([
            f"{h:02d}:{mi:02d}", f"{h:02d}.{mi:02d} น.", f"{h}:{mi:02d} น.", f"{h:02d}:{mi:02d}:{self._r(0, 59):02d}",
            f"{h:02d}.{mi:02d}-{(h + 1) % 24:02d}.{mi:02d} น.", f"{h % 12 or 12}:{mi:02d} {self._pick(['AM', 'PM'])}",
        ]))

    def code(self) -> str:
        up = lambda n: "".join(self._pick(string.ascii_uppercase) for _ in range(n))
        pats = [
            lambda: f"{up(3)}-{self.digits(4)}-{up(2)}",
            lambda: f"INV-{self._r(2020, 2026)}-{self.digits(5)}",
            lambda: f"{up(2)}{self.digits(6)}",
            lambda: f"PO{self.digits(8)}",
            lambda: f"{self._pick('กขคงจฉชญ')}{self._pick('กขคงจฉชญ')} {self._r(1, 9999)}",           # license plate
            lambda: f"{self._r(1, 9)}{self._pick('กขคงจฉชญ')}{self._pick('กขคงจฉชญ')} {self._r(1, 9999)}",
            lambda: f"v{self._r(0, 9)}.{self._r(0, 20)}.{self._r(0, 99)}",
            lambda: f"{self.digits(3)}-{self.digits(1)}-{self.digits(5)}-{self.digits(1)}",             # bank account
            lambda: f"{self.digits(5)}",                                                                # postal code
            lambda: f"{self.digits(10)}",
            lambda: f"#{self.digits(self._r(3, 6))}",
        ]
        return self._pick(pats)()

    def email(self) -> str:
        user = self._pick(self.en).lower() + self._pick(["", ".", "_", ""]) + self._pick(["", self.digits(self._r(1, 4)), self._pick(self.en).lower()[:4]])
        dom = self._pick(["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "example.co.th",
                          "company.com", "ku.ac.th", "chula.ac.th", "go.th", "or.th"])
        return f"{user}@{dom}"

    def url(self) -> str:
        return self._pick(["www.", "https://", "http://", ""]) + self._pick(self.en).lower() + self._pick([".com", ".co.th", ".go.th", ".net", ".org", ".ac.th"]) + self._pick(["", "/", "/" + self._pick(self.en).lower(), "/" + self.digits(4)])

    def measure(self) -> str:
        n = self._pick([str(self._r(1, 999)), f"{self._r(0, 99)}.{self._r(0, 9)}", f"{self._r(1, 99)},{self.digits(3)}"])
        return self.maybe_thai_digits(n) + self._pick(["", " "]) + self._pick(UNITS)

    def numeric(self) -> str:
        return self._pick([self.price, self.price, self.phone, self.id13, self.date, self.date, self.time,
                           self.code, self.email, self.url, self.measure, self.measure])()

    def address(self) -> str:
        parts = [self.maybe_thai_digits(self._pick([f"{self._r(1, 999)}", f"{self._r(1, 999)}/{self._r(1, 99)}", f"{self._r(1, 999)}/{self._r(1, 999)}/{self._r(1, 9)}"]))]
        if self._p(0.5):
            parts.append(self._pick(["หมู่ ", "หมู่ที่ ", "ม."]) + str(self._r(1, 20)))
        if self._p(0.6):
            parts.append(self._pick(["ถนน", "ถ.", "ซอย", "ซ."]) + self._pick(self.short_words))
        if self._p(0.7):
            parts.append(self._pick(["ตำบล", "ต.", "แขวง"]) + self._pick(self.short_words))
        if self._p(0.7):
            parts.append(self._pick(["อำเภอ", "อ.", "เขต"]) + self._pick(self.short_words))
        parts.append(self._pick(["จังหวัด", "จ.", ""]) + self._pick(PROVINCES))
        if self._p(0.5):
            parts.append(self.digits(5))
        return " ".join(parts)

    def form_line(self) -> str:
        label = self._pick(LABELS)
        value = self._pick([self.thai_name, self.thai_name, self.address, self.phone, self.email, self.date,
                            self.id13, self.price, self.thai_phrase, self.code, self.english_token])()
        sep = self._pick([" ", " ", " : ", ": ", " - ", "\t", " ....", "……"]).replace("\t", " ")
        if sep == " ....":
            return f"{label} {'.' * self._r(4, 12)}{value}{'.' * self._r(0, 10)}"
        return f"{label}{sep}{value}"

    def sentence_like(self) -> str:
        s = self._pick([self.thai_phrase, self.thai_phrase, lambda: self._pick(DOC_PHRASES)])()
        r = self.rng.random()
        if r < 0.15:
            s = self._pick(LIST_MARKERS) + " " + s
        elif r < 0.30:
            s += " " + self._pick(PARTICLES)
        elif r < 0.45:
            s = f"{s} ({self.thai_phrase(2)})"
        elif r < 0.55:
            s = f"“{s}”" if self._p(0.5) else f'"{s}"'
        elif r < 0.65:
            s += self._pick([",", ".", "?", "!", ":", ";", " ฯลฯ", "ๆ"])
        elif r < 0.8:
            s = f"{s} {self.numeric()}"
        return s

    def mixed(self) -> str:
        parts = []
        for _ in range(self._r(2, 4)):
            parts.append(self._pick([self.thai_phrase, self.thai_phrase, self.english_token, self.numeric])())
        if all(not any("฀" <= c <= "๿" for c in p) for p in parts):
            parts.insert(self._r(0, len(parts)), self.thai_phrase())
        return " ".join(parts)

    def natural_chunk(self) -> str:
        for _ in range(20):
            line = self._pick(self.natural)
            if len(line) < self.min_len:
                continue
            if len(line) > self.max_len:
                start = self._r(0, max(0, len(line) - self.max_len))
                # walk forward to a legal start position
                while start < len(line) and (line[start] in _FOLLOWERS or (start > 0 and line[start - 1] in _LEADING)):
                    start += 1
                line = line[start:]
            return line
        return self.thai_phrase()

    # ---- main loop
    def one(self) -> Optional[str]:
        r = self.rng.random()
        nw = self.natural_weight
        rest = 1.0 - nw
        if r < nw:
            s = self.natural_chunk()
        else:
            r = (r - nw) / rest
            if r < 0.28:
                s = self.thai_phrase()
            elif r < 0.48:
                s = self.sentence_like()
            elif r < 0.64:
                s = self.mixed()
            elif r < 0.78:
                s = self.numeric()
            elif r < 0.92:
                s = self.form_line()
            else:
                s = self.english_line()
        s = normalize(s)
        if len(s) > self.max_len:
            s = safe_cut(s, self.max_len)
        if len(s) < self.min_len:
            return None
        if any(c not in self.charset for c in s):
            return None
        return s

    def lines(self, count: int, dedupe: bool = True) -> Iterator[str]:
        seen = set()
        produced = 0
        attempts = 0
        while produced < count and attempts < count * 20:
            attempts += 1
            s = self.one()
            if s is None:
                continue
            if dedupe:
                if s in seen:
                    continue
                seen.add(s)
            produced += 1
            yield s


def build_corpus(
    count: int,
    out: str,
    text_files: Iterable[str] = (),
    words_path: Optional[str] = None,
    en_words_path: Optional[str] = None,
    charset_path: Optional[str] = None,
    charset_out: Optional[str] = None,
    min_len: int = 6,
    max_len: int = 60,
    seed: Optional[int] = None,
    natural_weight: float = 0.45,
    thai_digit_prob: float = 0.08,
    dedupe: bool = True,
) -> int:
    words = _load_lines(words_path or os.path.join(HERE, "dicts", "th.txt"))
    en_words = _load_lines(en_words_path or os.path.join(HERE, "dicts", "en.txt"))
    natural: List[str] = []
    for p in text_files:
        for line in _load_lines(p):
            line = normalize(line)
            if len(line) >= min_len:
                natural.append(line)
    charset = load_charset(charset_path)
    builder = CorpusBuilder(words, en_words, natural, charset, min_len, max_len, thai_digit_prob, natural_weight, seed)

    used = set()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf8", newline="\n") as f:
        for line in builder.lines(count, dedupe=dedupe):
            f.write(line + "\n")
            used.update(line)
            n += 1
    if charset_out:
        used.discard(" ")  # PaddleOCR: space comes from use_space_char, not the dict
        with open(charset_out, "w", encoding="utf8", newline="\n") as f:
            f.write("\n".join(sorted(used)) + "\n")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a mixed Thai/English/numeric corpus for recognizer training.")
    ap.add_argument("--count", type=int, default=100000, help="lines to generate")
    ap.add_argument("--out", default="out/corpus.txt")
    ap.add_argument("--charset_out", default=None, help="write the set of characters used (one per line), e.g. out/charset.txt")
    ap.add_argument("--text", nargs="*", default=[], help="natural Thai text file(s), one paragraph/sentence per line")
    ap.add_argument("--words", default=None, help="Thai word list (default trdg/dicts/th.txt)")
    ap.add_argument("--en_words", default=None, help="English word list (default trdg/dicts/en.txt)")
    ap.add_argument("--charset", default=None, help="restrict output to these characters (one per line)")
    ap.add_argument("--min_len", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--natural_weight", type=float, default=0.45, help="share of lines taken from --text files")
    ap.add_argument("--thai_digit_prob", type=float, default=0.08, help="probability a number is written in Thai digits")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_dedupe", action="store_true")
    a = ap.parse_args(argv)
    n = build_corpus(a.count, a.out, a.text, a.words, a.en_words, a.charset, a.charset_out,
                     a.min_len, a.max_len, a.seed, a.natural_weight, a.thai_digit_prob, not a.no_dedupe)
    print(f"wrote {n} lines -> {a.out}" + (f", charset -> {a.charset_out}" if a.charset_out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
