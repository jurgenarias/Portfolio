## Problem Statement

How to increase SAT participation in a particular state

### Executive Summary

We were given data for the 2017 and 2018 ACT and SAT and we manipulated it using pandas in order to make statistical analysis.

## Contents

- [Summary of project](#Summary-of-project)
- [Data Dictionary](#Data-Dictionary)
- [Conclusions](#Conclusions)
- [Sources](#Data-sources)


#### Summary of project:
We gathered and evaluated data about the SAT and ACT for 2017 and 2018 in order to effectively make recommendations to the College Board to improve our SAT participation in a particular state. We played a lot with our dataframe, visualization and statistics to make sense of all patterns and information in order to provide good recommendations.


#### Data Dictionary:
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**state**|*object*|SAT|State from where the results were recorded|
|**state**|*object*|SAT|State from where the results were recorded| 
|**sat_participation_2017**|*float*|SAT|Percentage of participation by state|
|**act_english_2017**|*float*|ACT|Average Scores for the English ACT section in 2017 by state|
|**act_math_2017**|*float*|ACT|Average Scores for the Math ACT section in 2017 by state|
|**act_reading_2017**|*float*|ACT|Average Scores for the Reading ACT section in 2017 by state|
|**act_science_2017**|*float*|ACT|Average Scores for the Science ACT section in 2017 by state|
|**act_composite_2017**|*float*|ACT|Average Composite Scores for the ACT in 2017 by state|


#### Conclusions:
We need to invest in New Mexico. I focused on New Mexico where the participation rate for ACT is relatively high for a state where the ACT is not required. The College Board should address New Mexico and talk to the schools about making the SAT their preferred test because as clearly outlined in the presentation, it has a good chance of improving scores for that particular population.


#### Data sources:
Source for SAT data: https://blog.collegevine.com/here-are-the-average-sat-scores-by-state/  
Source for ACT data: https://blog.prepscholar.com/act-scores-by-state-averages-highs-and-lows
Source for state test requirements: https://magoosh.com/hs/act/2017/states-that-require-the-act-or-sat/
Source for state test preference: https://upload.wikimedia.org/wikipedia/commons/b/bd/SAT-ACT-Preference-Map.svg
Source for article about sat vs act: https://www.usnews.com/education/best-colleges/articles/act-vs-sat-how-to-decide-which-test-to-take
Source for population of latinos in the US: https://www.worldatlas.com/articles/us-states-with-the-largest-relative-hispanic-and-latino-populations.html
Source for ACT scores in New Mexico: https://www.act.org/content/dam/act/unsecured/documents/cccr2017/P_32_329999_S_S_N00_ACT-GCPR_New_Mexico.pdf
Source for SAT scores in New Mexico: https://reports.collegeboard.org/pdf/2017-new-mexico-sat-suite-assessments-annual-report.pdf
Source for conversion table: https://www.act.org/content/dam/act/unsecured/documents/ACT-SAT-Concordance-Tables.pdf
