"""Field survey CSV lookup and human-readable date helpers for volume reports."""
import calendar
import datetime
import glob
import os
import re
from pathlib import Path

import numpy as np

from . import io

DEFAULT_FILENAME_DATE = r"(?P<yy>\d\d)-(?P<doy>\d{3})-"


def doy_date(year, doy):
    """Date for 1-based day-of-year `doy` in `year`."""
    return datetime.date(int(year), 1, 1) + datetime.timedelta(days=int(doy) - 1)


def parse_day(day, year):
    """Convert a day-of-year (int/str), ISO date ('YYYY-MM-DD'), or `datetime.date` -> `(date, yy, doy)`."""
    if isinstance(day, datetime.date) and not isinstance(day, datetime.datetime):
        d = day
    elif isinstance(day, int) or (isinstance(day, str) and day.strip().isdigit()):
        d = doy_date(int(year), int(day))
    elif isinstance(day, str):
        d = datetime.date.fromisoformat(day.strip())
    else:
        raise ValueError(f"unsupported day specifier: {day!r}")
    return d, d.year % 100, d.timetuple().tm_yday


def nice_date(d, year=True):
    """Format a date as 'Wed Nov 18, 2026' (or 'Wed Nov 18' when year=False)."""
    return f"{d:%a %b} {d.day}" + (f", {d.year}" if year else "")


def readable(text, year=None):
    """Rewrite YY-DDD day-of-year tokens in `text` (e.g. '26-292') as 'Oct 19'."""
    century = (int(year) // 100) * 100 if year is not None else 2000

    def _sub(m):
        y = century + int(m.group(1))
        doy = int(m.group(2))
        max_doy = 366 if calendar.isleap(y) else 365
        if doy < 1 or doy > max_doy:
            return m.group(0)
        d = doy_date(y, doy)
        return f"{d:%b} {d.day}"

    return re.sub(r"\b(\d{2})-(\d{3})\b", _sub, str(text))


def date_from_filename(name, pattern=DEFAULT_FILENAME_DATE, century=2000):
    """Extract a `datetime.date` from `name` using regex `pattern`, or None if no match."""
    m = re.search(pattern, os.path.basename(str(name)))
    if not m:
        return None
    gd = m.groupdict()
    if "year" in gd and "month" in gd and "day" in gd:
        return datetime.date(int(gd["year"]), int(gd["month"]), int(gd["day"]))
    if "yyyy" in gd and "doy" in gd:
        return doy_date(int(gd["yyyy"]), int(gd["doy"]))
    if "yy" in gd and "doy" in gd:
        return doy_date(century + int(gd["yy"]), int(gd["doy"]))
    return None


class FieldData:
    """Lookup shots and source filenames from dated field survey CSVs using a config dict.

    Config keys:
      glob           glob template with {yy}, {yyyy}, {doy}, {date}, {filter} placeholders
      filename_date  regex with named groups (yy+doy, yyyy+doy, or year+month+day)
      exclude        substrings (case-insensitive) to skip in matched paths
      filter         string substituted into {filter} in `glob`
      max_offset     maximum plan offset from alignment CL (m, default 5.0)
      order          CSV column order when headerless (default 'PNEZD')
    """

    def __init__(self, cfg, year, base_dir=None):
        self.cfg = dict(cfg or {})
        self.year = int(year)
        self.base_dir = Path(base_dir) if base_dir else None
        self.glob_tpl = self.cfg.get("glob", "")
        self.filename_date = self.cfg.get("filename_date", DEFAULT_FILENAME_DATE)
        self.exclude = tuple(k.upper() for k in self.cfg.get("exclude", ()))
        self.filter = self.cfg.get("filter", "")
        self.max_offset = float(self.cfg.get("max_offset", 5.0))
        self.order = self.cfg.get("order", "PNEZD")

    def _pattern(self, day):
        d, yy, doy = parse_day(day, self.year)
        pat = self.glob_tpl.format(yy=yy, yyyy=d.year, doy=doy, date=d.isoformat(), filter=self.filter)
        p = Path(pat)
        if not p.is_absolute() and self.base_dir is not None:
            pat = str(self.base_dir / pat)
        return pat, d

    def matching_files(self, day):
        """Sorted `[(path, date)]` of field CSVs matching `day` and not excluded."""
        if not self.glob_tpl:
            return []
        pat, fallback = self._pattern(day)
        century = (fallback.year // 100) * 100
        out = []
        for f in sorted(glob.glob(pat)):
            if self.exclude and any(k in f.upper() for k in self.exclude):
                continue
            fdate = date_from_filename(f, self.filename_date, century=century) or fallback
            out.append((f, fdate))
        return out

    def shots(self, specs):
        """Array `(n, 3)` of `(north, east, z)` shots matching `specs = [(day, code), ...]`."""
        chunks = []
        for day, code in specs:
            want = str(code).strip().upper()
            for f, _ in self.matching_files(day):
                sh = io.read_shots(f, order=self.order)
                if not len(sh):
                    continue
                mask = [c.strip().upper() == want for c in sh.code]
                if any(mask):
                    chunks.append(sh.points[mask])
        if not chunks:
            return np.zeros((0, 3), float)
        return np.vstack(chunks)

    def files_used(self, specs, s0, s1, corridor):
        """`[(filename, date)]` of field CSVs with matching shots inside stations `s0..s1` and `max_offset`."""
        hits = {}
        for day, code in specs:
            want = str(code).strip().upper()
            for f, fdate in self.matching_files(day):
                sh = io.read_shots(f, order=self.order)
                if not len(sh):
                    continue
                mask = [c.strip().upper() == want for c in sh.code]
                if not any(mask):
                    continue
                P = sh.points[mask]
                s, o = corridor.alignment.station_offset(P[:, 0], P[:, 1])
                if np.any((s >= s0 - 0.1) & (s <= s1 + 0.1) & (np.abs(o) < self.max_offset)):
                    hits[os.path.basename(f)] = fdate
        return sorted(hits.items(), key=lambda kv: (kv[1], kv[0]))
