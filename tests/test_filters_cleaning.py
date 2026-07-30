"""
Tests for the source cleanup layer in scripts/filters.py.

Fixtures are excerpts from the manual review pack at
data/review/writing-20260727T235857Z/raw/, trimmed to representative
problem patterns per source.
"""

import pytest

from scripts.build_writing_review_pack import _analyze_document
from scripts.filters import (
    _strip_report_toc,
    clean_document,
    is_prose,
    reflow_hard_wrapped,
    shingle_jaccard,
    strip_front_matter,
    strip_markdown_residue,
    strip_publisher_boilerplate,
    strip_web_chrome,
    wikimedia_is_article,
)


# =============================================================================
# Fixtures: real problem excerpts
# =============================================================================

# raw/gwern.txt, document 001 (https://gwern.net/speedrunning): nav junk,
# page-metadata block, bare TOC line-list, trailing link-dump/footer blocks.
GWERN_CHROME = """Skip to main content

On statistical and psychological grounds, we probably are not losing many future Einsteins to speedrunning or streaming or e-sports etc.

 2025-03-14–2025-04-09
 finished
 certainty: likely
 importance: 4

 similar
 bibliography

Base Rates

Randomness in Success

Regression to a Mean

Obsession As Key Input

External Links

From a statistical perspective, we can safely say that the probability that our hypothetical Sonic streamer would have been a noted entomologist is near zero, because the probability for any given person of becoming a noted anything is near zero.

Few people are remembered centuries later. Thousands of German men passed through academic life in that province of 1820s Germany without leaving any mark more notable than their payslips in the ledger-books. A major, successful video maker is a person who is one in millions.

Indeed, our video maker would not be so successful at videos either in another history of the world; the Salganik/Watts experimental results on randomizing media markets suggest that when an author like J. K. Rowling is number one in our world, they might be thousands of rungs down in another world. Random chance and network effects may elevate an ordinary, mediocre rival of little talent, and force him out of the video game, and the more randomness, the more likely the winners were ordinary people who are thus not wasted.

We can predict success for groups, but not individuals; forests, not trees. For individuals the base rates dominate everything else we might want to say about counterfactual careers.

Similar Links

 [Similar links by topic]

Bibliography

 [Bibliography of links/references used in page]

 [ Send Anonymous Feedback ]

[Quote Of The Day]

[Site Of The Day]

[Annotation Of The Day]

[adblock public service announcement]"""

# raw/gwern.txt, document 002 (https://gwern.net/static/build/Config/Utext.hs):
# Haskell source that slipped into a prose source.
GWERN_HASKELL = """{-# LANGUAGE OverloadedStrings #-}
-- | Utext: Markdown to Unicode-rich plain text configuration data & unit-tests
-- Author: gwern
-- License: CC-0

module Config.Utext where

import qualified Data.Text as T (Text)

-- | Truncation limit for trace/error messages showing input excerpts.
traceLimit :: Int
traceLimit = 80

data Style = Style
 { sBold :: !Bool
 , sItalic :: !Bool
 , sLigature :: !Bool
 } deriving (Eq, Show)

defaultStyle :: Style
defaultStyle = Style False False False"""

# raw/gwern.txt, document 008 (https://gwern.net/static/js/extracts-load.js):
# JavaScript fragment.
GWERN_JAVASCRIPT = """/*	This file should be loaded after all other extracts*.js files.
 */

Extracts.config = {
 /* Selector for containers within which targets may be found.
 */
 contentContainersSelector: [
 ".markdownBody",
 "#TOC",
 "#navbar"
 ].join(", "),

	/*	Selector for targets.
 */
	targetElementsSelector: [
 "a[href]"
	].join(", "),

	excludedElementsSelector: [
 ".section-self-link",
 ".footnote-self-link",
 "[aria-hidden='true']",
 "[href$='#top']"
	].join(", "),
}"""

# raw/gwern.txt, document 004
# (https://gwern.net/static/font/ss3/SourceSans3-BASIC-Black.otf): binary
# residue decoded with replacement characters.
GWERN_OTF_BINARY = (
    "OTTO � @CFF a(� � �~GDEF,�.U �� LGPOS"
    "DvLu � GSUB�3�i �0 tOS/2}r� �H `cmapNIi"
    " �� $head%�� �T 6hhea�� �$ $"
)

# raw/common_pile_doab.txt, document 001: IntechOpen publisher advertising
# block followed by real series prose.
DOAB_PUBLISHER_AD = """We are IntechOpen, the world's leading publisher of Open Access books Built by scientists, for scientists

7,000+ Open access books available

187,000+

International authors and editors

205M+ Downloads

156 Countries delivered to Our authors are among the

Top 1% most cited scientists

12.2%

Contributors from top 500 universities

Selection of our books indexed in the Book Citation Index in Web of Science™ Core Collection (BKCI)

### Interested in publishing with us? Contact book.department@intechopen.com

Numbers displayed above are based on latest data collected. For more information visit www.intechopen.com

### IntechOpen Book Series Agricultural Sciences Volume 8

### Aims and Scope of the Series

The importance of agriculture cannot be overstated. It helps sustain life, as it gives us the food we need to survive and provides opportunities for economic well-being. Agriculture helps people prosper around the world and combines the creativity, imagination, and skill involved in planting crops and raising animals with modern production methods and new technologies."""

# raw/common_pile_doab.txt, document 007 pattern: sub-1.5k-char orphan chunk
# (author line, received/accepted dates, bare abstract).
DOAB_ORPHAN_FRAGMENT = """M. A. Iqbal

Received: 12 January 2022 / Accepted: 3 March 2022

This chapter summarizes recent perspectives on grassland development. Grasslands cover a substantial fraction of the terrestrial surface and provide forage, biodiversity refuges, and carbon storage. The chapter reviews management options and outlines directions for future research on sustainable intensification."""

# raw/crs_reports.txt, document 001: PDF hard line-wrap with leading-space
# continuation lines, including across blank lines.
CRS_HARD_WRAPPED = """Legal Sidebar

Federal Appeals Court Finds That Probable Cause Is

 Required to Hold Aliens Pursuant to Immigration

 Detainers

09/08/2015

The recent decision by the U.S. Court of Appeals for the First Circuit in Morales v. Chadbourne could complicate

 current debates about “sanctuary cities,” a term which some use to describe jurisdictions that decline to honor

 immigration detainers. An immigration detainer is a document whereby U.S. Immigration and Customs Enforcement

 (ICE) requests that another law enforcement agency take certain actions, which can include holding an alien for up to

 48 hours after the alien would otherwise have been released so that ICE may assume custody.

Recent reports that an alien who shot and killed a woman after being released by San

 Francisco authorities had been the subject of an immigration detainer, which the San Francisco Sheriff declined to

 honor, have prompted some Members of Congress to propose measures that encourage states and localities to honor

 immigration detainers or otherwise cooperate in immigration enforcement."""

# raw/crs_reports.txt, USACE report: Contents/Figures/Tables listing dump
# before the summary prose (shown here post-reflow).
CRS_TOC_DUMP = """Contents

USACE Annual Appropriations

Trust Funds

USACE Supplemental Appropriations

Conclusion: Trends and Policy Questions

Figures

Figure 1. Budget Request and Annual Appropriations for USACE Civil Works, FY2005 Through FY2019

Figure 2. USACE Annual Appropriations, FY2005 Through FY2019

Tables

Table 1. Account Funding for USACE Supplemental Appropriations, FY1990-FY2018

Summary

The U.S. Army Corps of Engineers (USACE) is an agency within the Department of Defense with both military and civil works responsibilities. The agency's civil works activities consist largely of the planning, construction, and operation of water resource projects to maintain navigable channels, reduce flood and storm damage, and restore aquatic ecosystems."""

# raw/plos.txt, document 001 (doi:10.1371/journal.pone.0131916): DOI, title,
# author and affiliation run fused into the head of the body paragraph.
PLOS_FUSED_FRONT_MATTER = (
    "10.1371/journal.pone.0131916 PONE-D-15-10287 Research Article Shedding "
    "New Light on the 18th Dynasty Mummies of the Royal Architect Kha and His "
    "Spouse Merit Raffaella Bianucci 1 2 3 Michael E. Habicht 4 Stephen "
    "Buckley 5 Joann Fletcher 5 1 Department of Public Health and Paediatric "
    "Sciences, Legal Medicine Section, Laboratory of Physical Anthropology, "
    "University of Turin, Turin, Italy 2 Centre for Ecological and "
    "Evolutionary Synthesis (CEES), Department Biosciences, University of "
    "Oslo, Oslo, Norway 4 Institute of Evolutionary Medicine, University of "
    "Zurich, Zurich, Switzerland Mark Spigelman Editor Hebrew University, "
    "ISRAEL * E-mail: frank.ruehli@iem.uzh.ch 22 7 2015 2015 10 7 e0131916 9 "
    "3 2015 8 6 2015 The mummies of Kha and his wife Merit were found intact "
    "in an undisturbed tomb in western Thebes near the ancient workers’ "
    "village of Deir el-Medina. Previous MDCT investigations showed that the "
    "bodies of Kha and Merit did not undergo classical royal 18th Dynasty "
    "artificial mummification, which included removal of the internal organs. "
    "It was, therefore, concluded that the retention of the viscera in the "
    "body, combined with an absence of canopic jars in the burial chamber, "
    "meant the couple underwent a short and shoddy funerary procedure, "
    "despite their relative wealth at death. Nevertheless, all internal "
    "organs showed a very good state of preservation, which contradicts the "
    "previous interpretation above."
)

# Same front matter split into its own leading paragraphs.
PLOS_PARAGRAPH_FRONT_MATTER = """10.1371/journal.pone.0131916 PONE-D-15-10287 Research Article

Raffaella Bianucci 1 2 3 Michael E. Habicht 4 Stephen Buckley 5 Joann Fletcher 5 1 Department of Public Health and Paediatric Sciences, Laboratory of Physical Anthropology, University of Turin, Turin, Italy 2 Centre for Ecological and Evolutionary Synthesis, University of Oslo, Oslo, Norway 4 Institute of Evolutionary Medicine, University of Zurich, Zurich, Switzerland Mark Spigelman Editor Hebrew University, ISRAEL * E-mail: frank.ruehli@iem.uzh.ch

The mummies of Kha and his wife Merit were found intact in an undisturbed tomb in western Thebes near the ancient workers’ village of Deir el-Medina. In order to better understand the type of mummification used to embalm these bodies, both wrapped mummies were reinvestigated using new generation X-ray imaging and chemical microanalyses."""

# raw/common_pile_wikimedia.txt, document 003: talk page of bot notices.
WIKIMEDIA_TALK_PAGE = """Talk:Bath bus station

External links modified
Hello fellow Wikipedians,

I have just modified 2 one external links on Bath bus station. Please take a moment to review my edit. If you have any questions, or need the bot to ignore the links, or the page altogether, please visit this simple FaQ for additional information. I made the following changes:
 * Added archive https://web.archive.org/web/20080131165317/http://www.royalcrescentbath.com:80/HistoryBathatWar.htm to http://www.royalcrescentbath.com/HistoryBathatWar.htm
 * Added archive https://web.archive.org/web/20101117023638/http://bathnes.gov.uk/transportandstreets/travel/buses/pages/busstopmap.aspx to http://www.bathnes.gov.uk/transportandstreets/travel/buses/pages/busstopmap.aspx

Cheers.— InternetArchiveBot (Report bug) 08:27, 28 October 2016 (UTC)

External links modified
Hello fellow Wikipedians,

I have just modified one external link on Bath bus station. Please take a moment to review my edit.

Cheers.— InternetArchiveBot (Report bug) 20:27, 15 July 2017 (UTC)"""

# raw/common_pile_wikimedia.txt, document 002: template talk page with
# (UTC)-signed discussion.
WIKIMEDIA_TEMPLATE_TALK = """Template talk:AmericanTerrorism

Scope?
Would Timothy McVey or the Unabomber belong here? Or in a related template? How about the Fenians? Symbionese Liberation Army? Black Panthers? Klu Klux Klan? Geo Swan (talk) 21:52, 5 January 2009 (UTC)
 * As happened with CanadianTerrorism, I had two separate templates for WoT/non-WoT, and ended up merging them. That may not be the case here, but definitely start as a separate template. Sherurcij (speaker for the dead) 22:37, 5 January 2009 (UTC)"""

# raw/common_pile_wikimedia.txt, document 004: portal on-this-day bullet dump.
WIKIMEDIA_PORTAL = """Portal:Spaceflight/On This Day/28 October

 * 1964 - A Vostok-2 launches the Kosmos 50 satellite.
 * 1965 - A Voskhod rocket launches the Kosmos 94 satellite.
 * 1965 - A Thor-Agena launches a CORONA satellite.
 * 1966 - A Scout B launches the OV3-2 satellite.
 * 1967 - An R-36 launches the Kosmos 187 satellite.
 * 1971 - The last Black Arrow rocket makes Britain's first, and so far, only satellite launch, orbiting the Prospero X-3 satellite."""

# raw/common_pile_wikimedia.txt, document 008: AfD deletion-debate archive.
WIKIMEDIA_AFD = """Wikipedia:Articles for deletion/List of James Bond films with synopses

The result was delete. James Bond (film series) (now) has the same content. Sandstein 16:16, 26 October 2008 (UTC)

This is a redundant list, see List of James Bond films Lithoderm (talk) 23:18, 21 October 2008 (UTC)
 * Delete as redundant. Ten Pound Hammer and his otters (Broken clamshells) 23:24, 21 October 2008 (UTC)"""

# raw/common_pile_wikimedia.txt, document 010: mainspace title but a
# bullet-dominated track listing.
WIKIMEDIA_TRACK_LISTING = """Like Omigod! The 80s Pop Culture Box (Totally)

'Like Omigod! The 80s Pop Culture Box (Totally)' is a seven-disc, 142-track box set of popular music hits of the 1980s. Released by Rhino Records in 2002.

Disc 1 (21 tracks)

 * 1) "Whip It" — Devo 2:39 (1980)
 * 2) "Video Killed the Radio Star" (Single Version) — The Buggles 3:27 (1980)
 * 3) "Empire Strikes Back (Medley)" — Meco 3:03 (1980)
 * 4) "Another One Bites the Dust" — Queen 3:34 (1980)
 * 5) "Celebration" — Kool & the Gang 3:43 (1981)
 * 6) "The Breaks (Pt. 1)" — Kurtis Blow 4:09 (1980)
 * 7) "Let My Love Open the Door" — Pete Townshend 2:44 (1980)
 * 8) "Call Me" (Single Version) — Blondie 3:32 (1980)"""

# raw/common_pile_wikimedia.txt, document 009: real article prose (keep).
WIKIMEDIA_ARTICLE = """United States v. Crimmins

United States v. Crimmins, 123 F.2d 271 (2d Cir. 1941), was a case before the United States Court of Appeals for the Second Circuit about conspiracy to transport stolen securities in interstate commerce. John D. Crimmins, a lawyer practicing in Syracuse, New York, was convicted for his part in a conspiracy in which he bought stolen securities from an accomplice who also lived in New York. Crimmins appealed on the grounds that he did not know the bonds had been transported across state lines.

Judge Learned Hand wrote the court's opinion. He reasoned that a jury may have found Crimmins guilty of the substantive offense of using interstate commerce in the commission of a crime because he knowingly bought stolen bonds even if he didn't know where they were from. According to Hand, people can only be charged for the actions of co-conspirators that were mutually agreed upon.

In United States v. Feola (1975), the Supreme Court rejected Hand's analogy, holding that conspiracy to assault a federal agent required no greater mens rea than the substantive crime of assault."""

# raw/common_pile_wikimedia.txt, document 005: real article prose (keep).
WIKIMEDIA_ARTICLE_SHORT = """Twilight Comes Twice

Twilight Comes Twice is a children's book of free verse written by Ralph Fletcher and illustrated by Kate Kiesler. It was first published in 1997 and describes the transitions from night to day and from day to night.

Reception
Publishers Weekly said in their review: "In spite of the commanding beauty of the language and art, however, the book engages the reader's emotions only minimally. There are distinct pleasures to be had here, but they are chiefly cerebral." """

# raw/our_world_in_data.txt, energy country profiles: machine-templated pages
# identical except for the country name.
_OWID_COUNTRY_TEMPLATE = """Many of us want an overview of how much energy our country consumes, where it comes from, and if we’re making progress on decarbonizing our energy mix. This page provides the data for your chosen country across all of the key metrics on this topic.

In the selection box above you can also add or remove additional countries and they will appear on all of the charts on this page. This allows you to compare specific countries you might be interested in, and measure progress against others.

In the energy domain, there are many different units thrown around — joules, exajoules, million tonnes of oil equivalents, barrel equivalents, British thermal units, terawatt-hours, to name a few. This can be confusing, and make comparisons difficult. So at Our World in Data we try to maintain consistency by converting all energy data to watt-hours.

We will continue to update our data and charts with the latest global and country figures, typically on an annual basis.

See all Energy data for {country}

What is {country}’s average energy consumption per person?

When comparing the total energy consumption of countries, the differences often reflect variations in population size. It’s useful to look at differences in energy consumption per capita. This interactive chart shows the average energy consumption per person each year."""

OWID_CAPE_VERDE = _OWID_COUNTRY_TEMPLATE.format(country="Cape Verde")
OWID_NAMIBIA = _OWID_COUNTRY_TEMPLATE.format(country="Namibia")

# raw/our_world_in_data.txt, document 001: data insight followed by the
# site-wide feed appended to every page.
OWID_WITH_FEED = """COVID-19 was the third largest cause of death in 2021

According to the latest Global Burden of Disease Study — published last month by the Institute of Health Metrics and Evaluation (IHME) — COVID-19 was the third leading cause of death in 2021, after cardiovascular diseases and cancer.

These estimates suggest that COVID-19 was responsible for around eight million deaths in 2021. In many countries across South America and sub-Saharan Africa, the IHME reports that it was the leading cause of death.

Global improvements in healthcare have led to a steady reduction in the death rate from infectious diseases in recent decades, but the COVID-19 pandemic has reversed this trend.

Explore this data →

Related topic pages:

Causes of Death

COVID-19

 Copy link

Our latest Data Insights
See all Data Insights

July 25

Three-quarters of eggs produced in the United Kingdom are free-range

Growing up in the United Kingdom in the 1990s and early 2000s, almost all the eggs in our supermarket came from battery hens, those raised in small wire enclosures."""

# raw/standard_ebooks.txt, document 001 (Twenty Years After): clean prose
# that every cleaner must pass through unchanged.
CLEAN_PROSE = """In a splendid chamber of the Palais Royal, formerly styled the Palais Cardinal, a man was sitting in deep reverie, his head supported on his hands, leaning over a gilt and inlaid table which was covered with letters and papers. Behind this figure glowed a vast fireplace alive with leaping flames; great logs of oak blazed and crackled on the polished brass andirons whose flicker shone upon the superb habiliments of the lonely tenant of the room, which was illumined grandly by twin candelabra rich with wax lights.

Anyone who happened at that moment to contemplate that red simar⁠—the gorgeous robe of office⁠—and the rich lace, or who gazed on that pale brow, bent in anxious meditation, might, in the solitude of that apartment, combined with the silence of the antechambers and the measured paces of the guards upon the landing-place, have fancied that the shade of Cardinal Richelieu lingered still in his accustomed haunt.

It was, alas! the ghost of former greatness. France enfeebled, the authority of her sovereign contemned, her nobles returning to their former turbulence and insolence, her enemies within her frontiers⁠—all proved the great Richelieu no longer in existence."""

# raw/common_pile_gutenberg.txt, document 008 (The Man Who Wins): Gutenberg
# prose with 70-column hard wraps; the default pipeline must not touch it.
GUTENBERG_PROSE = """The Four Corners in Middleton made a pleasant drive from the
university town of Camberton. Many a time in the history of the house
a party of young fellows had driven over the old turnpike that started
where the arsenal used to stand in the sacred quarter of Camberton,
and as the evening sun gilded the low, fresh-water marshes beyond
Spring Pond, would trot on toward the rolling hills of Middleton.
After dinner, or a dance, or, perhaps, mere chat over a late supper,
they rode away at midnight singing as they whipped up their sleepy
nags and otherwise disturbing the decorum of night in Middleton.

The Ellwells had kept the old Four Corners in Middleton long after the
family had moved out into the wider world of Boston, and from farming
the land they had passed to summering on it, as the successive
generations went into the professions or into trade."""


# =============================================================================
# strip_web_chrome
# =============================================================================


class TestStripWebChrome:
    def test_removes_leading_nav_and_metadata(self):
        cleaned = strip_web_chrome(GWERN_CHROME)
        assert "Skip to main content" not in cleaned
        assert "certainty: likely" not in cleaned
        assert "importance: 4" not in cleaned
        assert "2025-03-14" not in cleaned

    def test_removes_bare_toc_line_list(self):
        cleaned = strip_web_chrome(GWERN_CHROME)
        assert "Base Rates" not in cleaned
        assert "Randomness in Success" not in cleaned
        assert "External Links" not in cleaned

    def test_removes_trailing_link_dump(self):
        cleaned = strip_web_chrome(GWERN_CHROME)
        assert "Similar Links" not in cleaned
        assert "[Quote Of The Day]" not in cleaned
        assert "Send Anonymous Feedback" not in cleaned

    def test_keeps_body_prose(self):
        cleaned = strip_web_chrome(GWERN_CHROME)
        assert "On statistical and psychological grounds" in cleaned
        assert "From a statistical perspective" in cleaned
        assert "forests, not trees" in cleaned

    def test_cuts_owid_feed(self):
        cleaned = strip_web_chrome(OWID_WITH_FEED)
        assert "reversed this trend" in cleaned
        assert "Our latest Data Insights" not in cleaned
        assert "Three-quarters of eggs" not in cleaned

    def test_clean_prose_unchanged(self):
        assert strip_web_chrome(CLEAN_PROSE) == CLEAN_PROSE


# =============================================================================
# strip_publisher_boilerplate
# =============================================================================


class TestStripPublisherBoilerplate:
    def test_removes_intechopen_ad_block(self):
        cleaned = strip_publisher_boilerplate(DOAB_PUBLISHER_AD)
        assert "We are IntechOpen" not in cleaned
        assert "205M+ Downloads" not in cleaned
        assert "187,000+" not in cleaned
        assert "Interested in publishing with us" not in cleaned
        assert "Top 1% most cited scientists" not in cleaned
        assert "For more information visit www.intechopen.com" not in cleaned

    def test_keeps_series_prose(self):
        cleaned = strip_publisher_boilerplate(DOAB_PUBLISHER_AD)
        assert "The importance of agriculture cannot be overstated" in cleaned

    def test_clean_prose_unchanged(self):
        assert strip_publisher_boilerplate(CLEAN_PROSE) == CLEAN_PROSE


class TestStripMarkdownResidue:
    def test_removes_markers(self):
        text = "### Heading\n\nThe protocol achieve**s** a 3<sup>rd</sup> level."
        cleaned = strip_markdown_residue(text)
        assert "###" not in cleaned
        assert "**" not in cleaned
        assert "<sup>" not in cleaned
        assert "achieves" in cleaned

    def test_clean_prose_unchanged(self):
        assert strip_markdown_residue(CLEAN_PROSE) == CLEAN_PROSE


# =============================================================================
# reflow_hard_wrapped
# =============================================================================


class TestReflowHardWrapped:
    def test_joins_indented_continuations(self):
        reflowed = reflow_hard_wrapped(CRS_HARD_WRAPPED)
        assert "could complicate current debates" in reflowed
        assert "Enforcement (ICE) requests" in reflowed
        assert "San Francisco authorities" in reflowed

    def test_joins_wrapped_title_lines(self):
        reflowed = reflow_hard_wrapped(CRS_HARD_WRAPPED)
        assert (
            "Federal Appeals Court Finds That Probable Cause Is Required to "
            "Hold Aliens Pursuant to Immigration Detainers" in reflowed
        )

    def test_preserves_paragraph_breaks(self):
        reflowed = reflow_hard_wrapped(CRS_HARD_WRAPPED)
        paragraphs = [p for p in reflowed.split("\n") if p.strip()]
        # Two body paragraphs stay separate
        assert any(p.startswith("The recent decision") for p in paragraphs)
        assert any(p.startswith("Recent reports") for p in paragraphs)

    def test_preserves_list_items(self):
        text = (
            "I made the following changes:\n"
            " * Added archive one\n"
            " * Added archive two\n"
            "\n"
            "Cheers."
        )
        reflowed = reflow_hard_wrapped(text)
        lines = reflowed.split("\n")
        assert "* Added archive one" in lines
        assert "* Added archive two" in lines

    def test_clean_prose_unchanged(self):
        assert reflow_hard_wrapped(CLEAN_PROSE) == CLEAN_PROSE


# =============================================================================
# strip_front_matter
# =============================================================================


class TestStripFrontMatter:
    def test_trims_fused_front_matter_prefix(self):
        cleaned = strip_front_matter(PLOS_FUSED_FRONT_MATTER)
        assert cleaned.startswith("The mummies of Kha")
        assert "10.1371" not in cleaned
        assert "University of Turin" not in cleaned
        assert "artificial mummification" in cleaned

    def test_drops_front_matter_paragraphs(self):
        cleaned = strip_front_matter(PLOS_PARAGRAPH_FRONT_MATTER)
        assert cleaned.startswith("The mummies of Kha")
        assert "Raffaella Bianucci" not in cleaned
        assert "frank.ruehli" not in cleaned

    def test_clean_prose_unchanged(self):
        assert strip_front_matter(CLEAN_PROSE) == CLEAN_PROSE


# =============================================================================
# _strip_report_toc
# =============================================================================


class TestStripReportToc:
    def test_drops_contents_and_figure_lists(self):
        cleaned = _strip_report_toc(CRS_TOC_DUMP)
        assert "Figure 1." not in cleaned
        assert "Table 1." not in cleaned
        assert "Trust Funds" not in cleaned
        assert "The U.S. Army Corps of Engineers" in cleaned

    def test_clean_prose_unchanged(self):
        assert _strip_report_toc(CLEAN_PROSE) == CLEAN_PROSE


# =============================================================================
# is_prose
# =============================================================================


class TestIsProse:
    def test_rejects_haskell_source(self):
        assert not is_prose(GWERN_HASKELL)

    def test_rejects_javascript_fragment(self):
        assert not is_prose(GWERN_JAVASCRIPT)

    def test_rejects_binary_residue(self):
        assert not is_prose(GWERN_OTF_BINARY)

    def test_rejects_empty(self):
        assert not is_prose("")
        assert not is_prose("   \n  ")

    def test_accepts_literary_prose(self):
        assert is_prose(CLEAN_PROSE)
        assert is_prose(GUTENBERG_PROSE)

    def test_accepts_report_prose(self):
        assert is_prose(reflow_hard_wrapped(CRS_HARD_WRAPPED))

    def test_accepts_wiki_article(self):
        assert is_prose(WIKIMEDIA_ARTICLE)


# =============================================================================
# wikimedia_is_article
# =============================================================================


class TestWikimediaIsArticle:
    @pytest.mark.parametrize(
        "text",
        [
            WIKIMEDIA_TALK_PAGE,
            WIKIMEDIA_TEMPLATE_TALK,
            WIKIMEDIA_PORTAL,
            WIKIMEDIA_AFD,
            WIKIMEDIA_TRACK_LISTING,
        ],
        ids=["talk", "template_talk", "portal", "afd", "track_listing"],
    )
    def test_rejects_non_article_pages(self, text):
        assert not wikimedia_is_article(text)

    def test_accepts_articles(self):
        assert wikimedia_is_article(WIKIMEDIA_ARTICLE)
        assert wikimedia_is_article(WIKIMEDIA_ARTICLE_SHORT)

    def test_rejects_template_braces(self):
        text = "Some page\n\n{{cite web}} {{stub}} {{infobox}} content here."
        assert not wikimedia_is_article(text)


# =============================================================================
# shingle_jaccard
# =============================================================================


class TestShingleJaccard:
    def test_country_template_pages_are_near_duplicates(self):
        similarity = shingle_jaccard(OWID_CAPE_VERDE, OWID_NAMIBIA)
        assert similarity >= 0.7
        assert similarity < 1.0

    def test_identical_text(self):
        assert shingle_jaccard(CLEAN_PROSE, CLEAN_PROSE) == 1.0

    def test_distinct_prose_is_not_flagged(self):
        assert shingle_jaccard(WIKIMEDIA_ARTICLE, WIKIMEDIA_ARTICLE_SHORT) < 0.3
        assert shingle_jaccard(CLEAN_PROSE, GUTENBERG_PROSE) < 0.3

    def test_empty_input(self):
        assert shingle_jaccard("", CLEAN_PROSE) == 0.0


# =============================================================================
# clean_document
# =============================================================================


class TestCleanDocument:
    def test_gwern_pipeline(self):
        cleaned = clean_document(GWERN_CHROME, "gwern")
        assert cleaned is not None
        assert "Skip to main content" not in cleaned
        assert "[Quote Of The Day]" not in cleaned
        assert "From a statistical perspective" in cleaned

    def test_gwern_drops_source_code(self):
        assert clean_document(GWERN_HASKELL, "gwern") is None
        assert clean_document(GWERN_JAVASCRIPT, "gwern") is None
        assert clean_document(GWERN_OTF_BINARY, "gwern") is None

    def test_doab_pipeline(self):
        cleaned = clean_document(DOAB_PUBLISHER_AD, "common_pile_doab")
        assert cleaned is None or "We are IntechOpen" not in cleaned

    def test_doab_drops_orphan_fragment(self):
        assert len(DOAB_ORPHAN_FRAGMENT) < 1500
        assert clean_document(DOAB_ORPHAN_FRAGMENT, "common_pile_doab") is None

    def test_crs_pipeline(self):
        cleaned = clean_document(CRS_HARD_WRAPPED, "crs_reports")
        assert cleaned is not None
        assert "could complicate current debates" in cleaned

    def test_plos_pipeline(self):
        cleaned = clean_document(PLOS_FUSED_FRONT_MATTER, "plos")
        assert cleaned is not None
        assert cleaned.startswith("The mummies of Kha")

    def test_wikimedia_drops_non_articles(self):
        assert clean_document(WIKIMEDIA_TALK_PAGE, "common_pile_wikimedia") is None
        assert clean_document(WIKIMEDIA_PORTAL, "common_pile_wikimedia") is None

    def test_wikimedia_keeps_articles(self):
        cleaned = clean_document(WIKIMEDIA_ARTICLE, "common_pile_wikimedia")
        assert cleaned is not None
        assert "Judge Learned Hand" in cleaned

    def test_default_pipeline_is_identity(self):
        assert clean_document(CLEAN_PROSE, "standard_ebooks") == CLEAN_PROSE
        assert clean_document(GUTENBERG_PROSE, "common_pile_gutenberg") == GUTENBERG_PROSE

    def test_drops_short_documents(self):
        assert clean_document("Too short to keep.", "standard_ebooks") is None

    def test_drops_empty(self):
        assert clean_document("", "gwern") is None
        assert clean_document("   ", "gwern") is None


# =============================================================================
# Triage flags (build_writing_review_pack._analyze_document)
# =============================================================================


def _make_document(text: str, source_id: str) -> dict:
    return {"source_id": source_id, "document_id": "test", "url": None, "text": text}


class TestTriageFlags:
    def test_near_duplicate_flag(self):
        first = _analyze_document(_make_document(OWID_CAPE_VERDE, "our_world_in_data"), 0, [])
        assert "near_duplicate" not in first["flags"]
        second = _analyze_document(
            _make_document(OWID_NAMIBIA, "our_world_in_data"), 0, [OWID_CAPE_VERDE]
        )
        assert "near_duplicate" in second["flags"]
        assert not second["automatic_artifact_clean"]

    def test_non_prose_flag(self):
        analysis = _analyze_document(_make_document(GWERN_HASKELL, "gwern"), 0, [])
        assert "non_prose" in analysis["flags"]
        clean = _analyze_document(_make_document(CLEAN_PROSE, "standard_ebooks"), 0, [])
        assert "non_prose" not in clean["flags"]

    def test_non_article_namespace_flag(self):
        analysis = _analyze_document(
            _make_document(WIKIMEDIA_TALK_PAGE, "common_pile_wikimedia"), 0, []
        )
        assert "non_article_namespace" in analysis["flags"]
        # Only applies to the wikimedia source
        other = _analyze_document(_make_document(WIKIMEDIA_TALK_PAGE, "gwern"), 0, [])
        assert "non_article_namespace" not in other["flags"]
