library(dplyr)
library(tidyverse)
library(modelsummary)

machine <- '/Users/michelev/Dropbox/lobbying'
ethics_subfolder <- file.path(machine, 'data/NYS_Ethics_Commission/')
legi_subfolder <- file.path(machine, 'data/NYS_LegiScan/')
df_subfolder <- file.path(machine, 'dataframes/')
tables_subfolder <- file.path(machine, 'tables/')

# Define columns to drop
columns_to_drop <- c('check_rule', 'full_name', 'full_name_with_nickname', 'full_name_wo_middle_name', 'full_name_with_alternative_nickname', 'possible_error', 'count_pieces', 'committee', 'suffix', 'first_name', 'last_name', 'middle_name', 'suffix', 'nickname')

# Read parties_df
parties_df <- read.csv(paste(ethics_subfolder, 'parties_lobbied_legislation_cleaned.csv', sep = ""), row.names = 1)
report_columns <- c('year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client', 'govt_body', 'activity', 'focus', 'party_name')

parties_df <- parties_df[!duplicated(parties_df[, report_columns]),]
parties_df$chamber_id <- as.integer(parties_df$chamber_id)
parties_df$filing_bimonth <- as.integer(parties_df$filing_bimonth)
parties_df$filing_semester <- as.integer(parties_df$filing_semester)
parties_df$contacted_staff_counsel <- as.logical(parties_df$contacted_staff_counsel)
parties_df$is_staff_counsel <- as.logical(parties_df$is_staff_counsel)
parties_df$entire_body <- as.logical(parties_df$entire_body)
parties_df$majority_body <- as.logical(parties_df$majority_body)
parties_df$people_id <- as.integer(parties_df$people_id)
parties_df$committee_id <- as.integer(parties_df$committee_id)
parties_df$is_matched_bill <- as.logical(parties_df$is_matched_bill)

parties_df <- parties_df[, !names(parties_df) %in% columns_to_drop]
parties_df <- parties_df[parties_df$is_matched_bill,]

see <- parties_df  %>%
  group_by(bill_number) %>%
  summarize(n_c = n_distinct(beneficial_client), n_p = n_distinct(people_id))

see <- parties_df  %>%
  group_by(people_id) %>%
  summarize(n_pl = n_distinct(principal_lobbyist), compensation = sum(total_compensation))

see$n_jl = see$n_pl/sum(see$n_pl)
plot(see$n_jl, log(see$compensation))

see <- parties_df  %>%
  group_by(people_id) %>%
  summarize(n_pl = n_distinct(principal_lobbyist), compensation = sum(total_compensation))

see$n_jl = see$n_pl/sum(see$n_pl)
plot(see$n_jl, log(see$compensation))

pivot_table_bill <- parties_df %>%
  group_by(principal_lobbyist, beneficial_client, bill_number) %>%
  summarize(distinct_activities = n_distinct(activity, na.rm = TRUE), distinct_parties = n_distinct(people_id, na.rm = TRUE))
