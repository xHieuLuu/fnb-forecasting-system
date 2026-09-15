from datetime import date

from fnb_forecast.data.calendar import build_vietnam_calendar


def test_vietnam_calendar_marks_tet_and_distance_features() -> None:
    frame = build_vietnam_calendar(date(2025, 1, 27), date(2025, 2, 2))

    tet = frame.loc[frame["date"].eq("2025-01-29")].iloc[0]

    assert tet["is_holiday"]
    assert tet["days_to_tet"] == 0
    assert frame["days_to_tet"].tolist() == [2, 1, 0, -1, -2, -3, -4]
