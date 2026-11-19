"""Tests for surfpatch.fielddata (dates, day-of-year conversion, filename regex, field CSV lookup)."""
import datetime

import numpy as np

from rxlkit.builder import build_rxl
from rxlkit.roadxml import Corridor
from surfpatch import fielddata, io


def test_doy_date_and_leap_years():
    assert fielddata.doy_date(2026, 1) == datetime.date(2026, 1, 1)
    assert fielddata.doy_date(2026, 60) == datetime.date(2026, 3, 1)
    assert fielddata.doy_date(2026, 365) == datetime.date(2026, 12, 31)
    # Leap year 2024
    assert fielddata.doy_date(2024, 60) == datetime.date(2024, 2, 29)
    assert fielddata.doy_date(2024, 366) == datetime.date(2024, 12, 31)


def test_nice_date():
    d = datetime.date(2026, 11, 18)
    assert fielddata.nice_date(d) == "Wed Nov 18, 2026"
    assert fielddata.nice_date(d, year=False) == "Wed Nov 18"
    d2 = datetime.date(2026, 1, 5)
    assert fielddata.nice_date(d2) == "Mon Jan 5, 2026"


def test_readable_descriptions_and_year_boundaries():
    assert fielddata.readable("26-292 / 26-299 FILL LIFT", 2026) == "Oct 19 / Oct 26 FILL LIFT"
    assert fielddata.readable("25-365 to 26-001", 2026) == "Dec 31 to Jan 1"
    assert fielddata.readable("24-060 / 24-366", 2024) == "Feb 29 / Dec 31"
    # Invalid day 366 in a non-leap year is left untouched
    assert fielddata.readable("26-366 stays", 2026) == "26-366 stays"


def test_date_from_filename():
    assert fielddata.date_from_filename("26-292-ABC-TRENCH.csv") == datetime.date(2026, 10, 19)
    assert fielddata.date_from_filename("24-060-LEAP.csv") == datetime.date(2024, 2, 29)
    assert fielddata.date_from_filename("no_date_here.csv") is None
    iso_pat = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    assert fielddata.date_from_filename("2026-10-19-survey.csv", iso_pat) == datetime.date(2026, 10, 19)


def test_field_data_shots_and_files_used(tmp_path):
    N0, E0 = 1000.0, 2000.0
    rxl_path = tmp_path / "al.rxl"
    rxl_path.write_text(build_rxl({
        "name": "AL",
        "horizontal": {"pis": [{"north": N0, "east": E0}, {"north": N0 + 100.0, "east": E0}]},
    }), encoding="utf-8")
    cor = Corridor.load(rxl_path)

    sub = tmp_path / "data" / "10-OCT" / "CREW"
    sub.mkdir(parents=True)
    # Matching file inside stations 10..30
    io.write_pnezd(sub / "26-292-AB-TRENCH.csv",
                   [[N0 + 15.0, E0 + 0.5, 100.5], [N0 + 20.0, E0 - 0.5, 100.6]], code="LIFT1")
    # Excluded file (contains ORIG)
    io.write_pnezd(sub / "26-292-AB-TRENCH-ORIG.csv",
                   [[N0 + 15.0, E0 + 0.0, 99.0]], code="LIFT1")
    # Matching file on another day, outside stations 10..30
    io.write_pnezd(sub / "26-299-AB-TRENCH.csv",
                   [[N0 + 80.0, E0 + 0.0, 101.0]], code="LIFT1")

    fd = fielddata.FieldData({
        "glob": "data/*/CREW/{yy}-{doy:03d}-*{filter}*.csv",
        "exclude": ["ORIG", "UTIL"],
        "filter": "TRENCH",
        "max_offset": 5.0,
    }, year=2026, base_dir=tmp_path)

    # Accepts both day-of-year and ISO date
    pts_doy = fd.shots([(292, "LIFT1")])
    pts_iso = fd.shots([("2026-10-19", "LIFT1")])
    assert len(pts_doy) == 2
    assert np.allclose(pts_doy, pts_iso)

    used = fd.files_used([(292, "LIFT1"), (299, "LIFT1")], 10.0, 30.0, cor)
    assert used == [("26-292-AB-TRENCH.csv", datetime.date(2026, 10, 19))]
