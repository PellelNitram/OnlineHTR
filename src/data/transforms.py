# Add transforms for datasets here.

# Datasets will return dicts with keys x, y, (optionally) t, stroke_nr, label, sample_name.

# Transforms that I need for sure:
# - As input a stack of (x, y, stroke_nr) and as output label.
# - same as above but with t
# - same as above but with transforms like differences, differences after equidistance transform, Bezier curves
