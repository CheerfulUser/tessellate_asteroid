"""All prose shown on the landing page. Edit this file, then run shape/build_landing.py.

This is plain text, NOT an f-string, so braces need no escaping and a typo cannot break the
build the way it could when the copy was embedded alongside the CSS and JS.

Numbers come from the catalogue at build time via {placeholders}, so they can never go stale:

    {n}        objects with a reliable period            {shapes}   with a convex shape model
    {new}      with no previously published period       {noshape}  with a lightcurve but no model
    {long}     with a period longer than 24 hours        {pages}    with a page (all of them)
    {base}     median observing baseline, in days        {med}      median period, hours
    {pts}      median measurements per object            {rots}     median rotations observed
    {lcdb}     with a published LCDB period

Write them as {n} or {base:.0f}; the usual Python format specs work. HTML entities such as
&mdash; and &rarr; are fine, and inline tags like <strong> or <em> are allowed.
"""

# The small uppercase line above the title.
EYEBROW = 'TESS sectors 27&ndash;55 &middot; TESSELLATE'

# The page title, and the browser tab title.
TITLE = 'Asteroid rotation & shape catalog'
TAB_TITLE = 'TESSELLATE asteroid catalog'

# The opening paragraph, directly under the title.
LEDE = """Rotation periods, phase-folded lightcurves and convex shape models for asteroids
observed by TESS. Since TESS observes the same field continuously at high cadence for at least a month, these
lightcurves can run unbroken for weeks &mdash; allowing us to recover precise rotation periods for both fast 
and slow rotators that are a challenge to recover from the ground."""

# The five headline numbers. Each entry is (value, label); the value may use {placeholders}.
KPIS = [
    ('{n:,}',          'rotation periods'),
    ('{new:,}',        'with no previously published period'),
    ('{long:,}',       'longer than 24 hours'),
    ('{base:.0f} days', 'median observing baseline'),
    ('{pts:,}',        'median measurements each'),
]

SEARCH_HEADING = 'Search the catalog'
SEARCH_NOTE = """Type a name or number for a quick look, or open one at random. Every object has
a page: {shapes:,} carry a convex shape model, and {noshape:,} have a period and folded
lightcurve but no model, each stating why."""
SEARCH_PLACEHOLDER = 'e.g. Eurydike, 3550, Lanzia&hellip;'
RANDOM_BUTTON = 'Random asteroid'
BROWSE_LINK = 'Browse the full catalog &rarr;'
BROWSE_NOTE = 'all {n:,} objects, with filters and sorting.'

POPULATION_HEADING = 'The population'

# Figure captions, keyed by the image file they belong to.
CAPTIONS = {
    'period_hist.png': """Rotation periods: {long:,} objects turn more slowly than once a day,
        the regime ground-based photometry struggles to close.""",
    'spin_size.png': """Spin rate vs size: Only 21 objects sit above the 2.2 hour spin barrier is where a loosely bound
        rubble pile would fly apart.""",
    'amplitude_period.png': """Amplitude vs period: Larger amplitudes imply more extreme aspect ratios.""",
    'orbital_elements.png': """Orbital parameters: The dashed lines are Jupiter
        mean-motion resonances, which clear the Kirkwood gaps; clumps are collisional families.
        The catalog samples the inner, middle and outer belt about equally.""",
}

DOWNLOAD_HEADING = 'Available data products'
DOWNLOAD_NOTE = """Each object page carries its phase-folded lightcurve with per-bin
uncertainties, the full stacked photometry behind it, and a rotatable convex shape model
&mdash; all downloadable, and the shape model is synchronised to the lightcurve so you can watch
which face produces which feature."""

# Where the photometry comes from, shown above the credit section.
# Citations verified against ADS; each bibcode was resolved through the ADS link
# gateway to a live DOI, so the references are confirmed rather than recalled.
DATA_HEADING = 'Data and photometry'
DATA = """Every lightcurve here is measured from TESS full-frame images by
<strong>TESSELLATE</strong>, an untargeted time domain search across
the TESS archive
(<a href="https://ui.adsabs.harvard.edu/abs/2025AJ....170..186R/abstract">Roxburgh et al. 2025</a>). 
TESSELLATE builds on the difference-imaging reduction in <strong>TESSreduce</strong>
(<a href="https://arxiv.org/abs/2111.15006">arXiv:2111.15006">Ridden-Harper et al.
2021). Because TESS revisits the same field continuously, the resulting photometry is well
suited to rotation periods: the sampling is near uniform and the coverage runs for weeks rather than
hours."""

# Credit for the inversion method and code, shown just above the caveat box.
# Citations verified against ADS; each bibcode was resolved through the ADS link
# gateway to a live DOI, so the references are confirmed rather than recalled.
CREDIT_HEADING = 'Shape models'
CREDIT = """Shape models are produced by convex lightcurve inversion using
<strong>convexinv</strong>, the implementation distributed with DAMIT. The method was developed
by Kaasalainen &amp; Torppa
(<a href="https://ui.adsabs.harvard.edu/abs/2001Icar..153...24K/abstract">2001</a>) 
and Kaasalainen, Torppa &amp; Muinonen 
(<a href="https://ui.adsabs.harvard.edu/abs/2001Icar..153...37K/abstract">2001</a>). 
The code and the Minkowski reconstruction that turns the fitted facet areas into a
closed polyhedron are distributed through <strong>DAMIT</strong>, the Database of Asteroid
Models from Inversion Techniques
(<a href="https://ui.adsabs.harvard.edu/abs/2010A&A...513A..46D/abstract">Ďurech, Sidorin
&amp; Kaasalainen 2010</a>). The rotation periods, the reliability
classification and the shape fits presented here are produced by TESSELLATE; the inversion method and code are
not."""

# The amber box at the foot of the page.
CAVEAT_TITLE = 'Reading a shape model.'
CAVEAT = """Most of these asteroids were seen during a single apparition, from one direction.
That constrains the shape but not its orientation in space, so the spin axis is assumed rather
than measured, as indicated on each page. Convex inversion also cannot represent concavities, so
models with sharp or complex minima fail to generate a shape model with convex. The rotation periods, by contrast, are
well determined, and often more precise than the Light Curve Data Base (LCDB)."""

# ---- the browse page ----
BROWSE_TAB_TITLE = 'Browse the TESSELLATE asteroid catalog'
BROWSE_TITLE = 'Browse the catalog'
BROWSE_HOME_LINK = '&larr; Catalog home'
BROWSE_SEARCH_PLACEHOLDER = 'Search {n:,} objects by name or number&hellip;'
