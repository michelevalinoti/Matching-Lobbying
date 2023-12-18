library(dplyr)
library(tidyverse)
library(stringr)
library(modelsummary)
library(stargazer)
library(ggplot2)
library(tidyr)
library(purrr)

machine <- '/Users/michelev/Dropbox/lobbying'
ethics_subfolder <- file.path(machine, 'data/NYS_Ethics_Commission/')
legi_subfolder <- file.path(machine, 'data/NYS_LegiScan/')
df_subfolder <- file.path(machine, 'dataframes/')
tables_subfolder <- file.path(machine, 'tables/')
figures_subfolder <- file.path(machine, 'figures/')
latex_tables_subfolder <- file.path(machine, 'latex_tables/')
# Define columns to drop
columns_to_drop <- c('check_rule', 'full_name', 'full_name_with_nickname', 'full_name_wo_middle_name', 'full_name_with_alternative_nickname', 'possible_error', 'count_pieces', 'committee', 'suffix', 'first_name', 'last_name', 'middle_name', 'suffix', 'nickname')

# Read parties_df
parties_df <- read.csv(paste(ethics_subfolder, 'parties_lobbied_legislation_cleaned.csv', sep = ""), row.names = 1)
parties_df <- parties_df %>%
  separate_rows(beneficial_client, sep = "\n+") %>%
  # If you want to remove leading/trailing whitespace in the values, you can use trimws
  mutate(beneficial_client = trimws(beneficial_client))
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
#parties_df <- parties_df[parties_df$is_matched_bill,]

# Define columns to keep for bills_df
#columns_to_keep <- c('bill_id', 'session_id', 'bill_number', 'bill_type', 'bill_type_id', 'body', 'body_id')
# Read bills_df
bills_df <- read.csv(paste(legi_subfolder, 'bills_details_all.csv', sep = ""))
#bills_df <- bills_df[, columns_to_keep]
# Restrict to bill (exclude resolutions)
bills_df <- bills_df[bills_df$bill_type_id ==1,]
bills_df <- bills_df %>%
  filter(session_id %in% c(1644,1813))

# Read history_df
history_df <- read.csv(paste(legi_subfolder, 'history_all.csv', sep = ""))
colnames(history_df)[colnames(history_df) == 'name'] <- 'committee_name'
history_df$date <- as.Date(history_df$date)
history_df$year <- as.integer(format(history_df$date, '%Y'))
history_df$month <- as.integer(format(history_df$date, '%m'))
history_df$day <- as.integer(format(history_df$date, '%d'))
history_df$date_number <- history_df$year + as.numeric(format(history_df$date, '%j')) / 365
history_df$event_rank <- ave(history_df$date_number, history_df$bill_id, FUN = rank)
history_df$event_rank <- as.integer(history_df$event_rank)


event_table  <- read.csv(paste(tables_subfolder, 'event_table.csv', sep = ""))
contact_grouped <- read.csv(paste(tables_subfolder, 'contact_grouped.csv', sep = ""))

sponsors_contacts_table <- read.csv(paste(tables_subfolder, 'sponsors_contacts_table.csv', sep = ""))




filtered_table <- history_df %>%
  filter(year >= 2019 & year <= 2022) %>%  # Filter for the desired years
  group_by(bill_id) %>%  # Group by bill_id
  filter(sum(event == 9) == 1 & all(event %in% c(1, 9))) %>%
  left_join(bills_df[, c("bill_id", "session_id", "bill_number")], by = c("bill_id", "session_id"))

parties_summary <- parties_df %>%
  filter(retained_flag == "True") %>%
  group_by(bill_number, session_id) %>%
  summarise(distinct_lobbyists = n_distinct(principal_lobbyist), distinct_client = n_distinct(beneficial_client), distinct_people = n_distinct(people_id))

# Left join parties_summary with filtered_table based on bill_id
result_df <- left_join(parties_summary, filtered_table, by = c("session_id", "bill_number"))

filtered_table <- filtered_table %>%
  group_by()
lowest_rank <- history_df %>%
  group_by(bill_id) %>%
  slice_min(order_by = event_rank, n = 1) %>%
  rename(date_introduced = date, date_number_introduced = date_number)
lowest_rank_referral <- history_df %>%
  filter(event == 9) %>%
  group_by(bill_id) %>%
  slice_min(order_by = event_within_rank, n = 1) %>%
  rename(date_first_referral = date, date_number_first_referral = date_number, committee_id_introduced = committee_id, chamber_id_introduced = chamber_id)
highest_rank <- history_df %>%
  group_by(bill_id) %>%
  slice_max(order_by = event_rank, n = 1) %>%
  rename(date_last_updated = date, date_number_last_updated = date_number)

history_df <- history_df %>%
  left_join(lowest_rank %>% select(bill_id, date_introduced, date_number_introduced),
            by = "bill_id") %>%
  distinct(bill_id, date, event, committee_id, chamber_id, .keep_all = TRUE) %>%
  left_join(lowest_rank_referral %>% select(bill_id, date_first_referral, date_number_first_referral, committee_id_introduced, chamber_id_introduced),
            by = "bill_id") %>%
  distinct(bill_id, date, event, committee_id, chamber_id, .keep_all = TRUE) %>%
  left_join(highest_rank %>% select(bill_id, date_last_updated, date_number_last_updated),
            by = "bill_id") %>%
  distinct(bill_id, date, event, committee_id, chamber_id, .keep_all = TRUE)
  
# Find the date of the following referral (if any)
# Function to find the 'next_referral'
find_next_referral <- function(event_col) {
  next_event <- event_col -1
  next_event[next_event >= max(event_col, na.rm = TRUE)] <- NA
  
  return(next_event)
}

# Apply the function to each 'bill_id' group and left join it to the original dataframe
next_rank <- history_df %>%
  filter(event == 9) %>%
  group_by(bill_id) %>%
  mutate(event_within_rank = find_next_referral(event_within_rank)) %>%
  rename(next_referral = event_within_rank, date_next_referral = date, date_number_next_referral = date_number, committee_id_next_referral = committee_id, chamber_id_next_referral = chamber_id) %>%
  ungroup() %>%
  right_join(., history_df %>% filter(event == 9) %>% select(bill_id, event_within_rank, date, date_number, committee_id, chamber_id), by = c("bill_id", "next_referral" = "event_within_rank")) 

history_df <- history_df %>%
  left_join(next_rank %>% select(bill_id, date, event, committee_id, chamber_id, date_next_referral, date_number_next_referral, committee_id_next_referral, chamber_id_next_referral),
            by = c("bill_id", "date", "event", "committee_id", "chamber_id")) %>%
  distinct(bill_id, date, event, committee_id, chamber_id, .keep_all = TRUE)

# Read sponsors_df
sponsors_df <- read.csv(paste(legi_subfolder, 'sponsors_all.csv', sep = ""))
sponsors_df <- sponsors_df[, !names(sponsors_df) %in% c('name', 'committee_id', 'committee_sponsor')]



summary_history <- history_df %>%
  group_by(bill_id) %>%
  summarize(time_length = max(date_number_last_updated - date_number_introduced),
            time_length_referral = mean(date_number_next_referral - date_number, na.rm = TRUE),
            passed = sum(event == 4, na.rm = TRUE),
            number_referrals = sum(event == 9, na.rm = TRUE),
            number_distinct_committees = n_distinct(committee_id, na.rm = TRUE),
            number_distinct_chambers = n_distinct(chamber_id, na.rm = TRUE))



summary_sponsors <- sponsors_df %>%
  group_by(bill_id) %>%
  summarize(count_sponsors = n(),
            count_sponsors_main = sum(sponsor_type_id == 1, na.rm = TRUE),
            count_sponsors_co = sum(sponsor_type_id == 2, na.rm = TRUE))
  

stats_bills <- bills_df %>%
  mutate(senate_bill = ifelse(body_id == 74, 1, 0))

stats_bills <- left_join(stats_bills, summary_history, on = 'bill_id')


stats_bills <- left_join(stats_bills, summary_sponsors, on = 'bill_id')

stats_bills <- left_join(stats_bills, parties_df %>% select(session_id, bill_number, is_matched_bill) %>%
                   distinct(bill_number, session_id, .keep_all = TRUE), on = c('session_id', 'bill_number'))
#stats_bills$is_matched_bill <- ifelse(stats_bills$is_matched_bill == TRUE, TRUE, FALSE)
stats_bills <- stats_bills %>%
  mutate(is_matched_bill = ifelse(is_matched_bill == FALSE | is.na(is_matched_bill), FALSE, TRUE))

covariates <- c('senate_bill', 'passed', 'time_length', 'number_referrals', 'number_distinct_committees', 'number_distinct_chambers', 'time_length_referral', 'count_sponsors', 'count_sponsors_main', 'count_sponsors_co', 'is_matched_bill')


datasummary(
  All(stats_bills[covariates]) ~ (1 + (is_matched_bill==TRUE)) * (N + Mean + SD + Min + Max),
  data = stats_bills[covariates],output = paste(latex_tables_subfolder, 'summary_bills.tex', sep = ""))

new_df <- parties_df %>% 
  left_join(., bills_df, by = c('session_id', 'bill_number'), suffix = c("_report", "_bill"))
new_df <- new_df %>% 
   left_join(., history_df[complete.cases(history_df[, c('session_id', 'bill_id', 'committee_id')]),], by = c('session_id', 'bill_id', 'committee_id'), suffix = c("_report", "_progress"))
new_df <- new_df %>% 
  left_join(., sponsors_df, by = c('session_id', 'bill_id', 'people_id'), suffix = c("_report", "_sponsor"))
  
write.csv(new_df, paste(df_subfolder, "reports_merged.csv", sep=""), row.names=FALSE)
new_df$filing_bimonth <- NA
new_df$filing_bimonth[new_df$filing_type %in% 'BI-MONTHLY'] <- fromFilingPeriodToBimonth(new_df$filing_period[new_df$filing_type %in% 'BI-MONTHLY'])
new_df$filing_semester <- NA
new_df$filing_semester[new_df$filing_type %in% 'BI-MONTHLY'] <- (new_df$filing_bimonth[new_df$filing_type %in% 'BI-MONTHLY'] / 3 > 1) + 1


# import data frame with legislative bills only
#file_path <- file.path(table_subfolder, 'final_table.csv')
#df <- read.csv(file_path)

# restrict where both the bill and the person is known (i.e., is a person)
data_contacts <-new_df[complete.cases(new_df$people_id, new_df$bill_id), ]
report_columns_bill <- c('session_id', 'year_report', 'filing_bimonth', 'bill_id', 'people_id', 'principal_lobbyist', 'beneficial_client')
# delete duplicate rows (if any)
data_contacts <- data_contacts[!duplicated(data_contacts[, report_columns_bill]),]

data_contacts$lobbyist_client <- paste(data_contacts$principal_lobbyist, data_contacts$beneficial_client, sep = "_")
data_contacts$year_bimonth <- paste(data_contacts$year_report, data_contacts$bimonth, sep = "-")
# define variables (sponsor, sponsor of type 1, sponsor of type 2)
data_contacts$sponsor_type_id <- ifelse(is.na(data_contacts$sponsor_type_id), 0, data_contacts$sponsor_type_id)
data_contacts$is_sponsor <- ifelse(data_contacts$sponsor_type_id == 0, 0, 1)
data_contacts$is_sponsor_rank_1 <- ifelse(data_contacts$sponsor_type_id == 1, 1, 0)
data_contacts$is_sponsor_rank_2 <- ifelse(data_contacts$sponsor_type_id == 2, 1, 0)


createOLSDataFrame <- function(data_contacts, group_cols, cols_tot) {
  df_grouped <- data_contacts %>% 
                group_by(across(all_of(group_cols))) %>%
                summarise(contacts_individual = n_distinct(.data[[group_count_cols]]))
  group_cols_tot <- group_cols[group_cols != 'people_id']
  df_grouped_tot <- data_contacts %>% 
                    group_by(across(all_of(group_cols_tot))) %>%
                    summarise(contacts_total = n_distinct(.data[[group_count_cols]]))
  
  keep_cols <- c(group_cols, c('session_id', 'is_sponsor', 'is_sponsor_rank_1', 'is_sponsor_rank_2', 'sponsor_type_id'))
  df_reg <- left_join(df_grouped, data_contacts[, keep_cols], by = group_cols)
  df_reg <- left_join(df_reg, df_grouped_tot, by = group_cols_tot)
  df_reg$shares <- df_reg$contacts_individual/(max(df_reg$contacts_total))
  df_reg$shares_0 <- (max(df_reg$contacts_total)-df_reg$contacts_individual)/(max(df_reg$contacts_total))
  #cond = (df_reg$shares >= 0.001) & (df_reg$shares <= 0.999)
  cond = df_reg$contacts_total > 25
  df_reg <- df_reg[cond,]
  df_reg$log_shares <- log(df_reg$shares)
  df_reg$log_shares_0 <- log(df_reg$shares_0)
  df_reg$log_shares_diff <- df_reg$log_shares - df_reg$log_shares_0
  return (df_reg)
}
  
  
group_cols <- c('session_id', 'people_id', 'bill_id')
group_count_cols <- 'lobbyist_client'

df_reg <- createOLSDataFrame(data_contacts, group_cols, cols_tot)
model <- lm(log_shares_diff ~ is_sponsor + 0 + factor(session_id) + factor(bill_id) + factor(people_id), data = df_reg)
summary(model)

model <- lm(log_shares ~ is_sponsor_rank_1 + is_sponsor_rank_2 + 0 + factor(people_id), data = df_reg)
summary(model)


# market = bill
# choice = public officials
group_cols = c('people_id', 'bill_id')
group_tot_cols = c('principal_lobbyist', 'beneficial_client')

# market = [bill, bimonth]
# choice = public officials
group_cols = c('people_id', 'bill_id', 'filing_bimonth')
group_tot_cols = c('principal_lobbyist', 'beneficial_client')

# market = [bill, lobbyist]
# choice = public officials
group_cols = c('people_id', 'bill_id', 'principal_lobbyist')
group_tot_cols = c('beneficial_client')

df_reg <- createOLSDataFrame(data_contacts, group_cols, cols_tot)
model <- lm(log_shares ~ is_sponsor + 0 + factor(people_id) + factor(principal_lobbyist), data = df_reg)
summary(model)

model <- lm(log_shares ~ is_sponsor_rank_1 + is_sponsor_rank_2 + 0 + factor(people_id) + factor(principal_lobbyist), data = df_reg)
summary(model)

group_cols = c('session_id', 'bill_id', 'people_id', 'principal_lobbyist', 'beneficial_client')
group_tot_cols = c('year_bimonth')
df_reg <- createOLSDataFrame(data_contacts, group_cols, cols_tot)
model <- lm(log_shares ~ is_sponsor + 0 + factor(people_id), data = df_reg)
summary(model)

# market = [bill, client]
# choice = public officials
group_cols = c('people_id', 'bill_id', 'beneficial_client')
group_tot_cols = c('principal_lobbyist')


keep_cols = c('market_id',
              'noalt',
              'session_id',
              'year',
              #'principal_lobbyist',
              #'beneficial_client',
              'contacted_staff_counsel',
              'is_staff_counsel',
              'people_id',
              'bill_id',
              'sponsor_type_id',
              'is_sponsor',
              'is_sponsor_rank_1',
              'is_sponsor_rank_2')


event_table  <- read.csv(paste(tables_subfolder, 'event_table.csv', sep = ""))
contact_grouped <- read.csv(paste(tables_subfolder, 'contact_grouped.csv', sep = ""))
sponsors_contacts_table <- read.csv(paste(tables_subfolder, 'sponsors_contacts_table.csv', sep = ""))
sponsors_clients_lobbyist_contacts_table <- read.csv(paste(tables_subfolder, 'sponsors_clients_lobbyist_contacts_table.csv', sep = ""))
authors_table <- read.csv(paste(tables_subfolder, 'authors_table.csv', sep = ""))
committees_table <- read.csv(paste(tables_subfolder, 'committees_table.csv', sep = ""))
committee_members <- read.csv(paste(legi_subfolder, 'committees_members_all_manual.csv', sep = ""))
people_all <- read.csv(paste(legi_subfolder, 'people_all.csv', sep = ""))
intensity_lobbying  <- read.csv(paste(tables_subfolder, 'intensity_lobbying_before.csv', sep = ""))
bills_conditions  <- read.csv(paste(tables_subfolder, 'bills_conditions.csv', sep = ""))
contacts_history  <- read.csv(paste(tables_subfolder, 'contacts_history.csv', sep = ""))
contacts_history_avg  <- read.csv(paste(tables_subfolder, 'contacts_history_avg.csv', sep = ""))
people_session_contacts  <- read.csv(paste(tables_subfolder, 'people_session_contacts_2.csv', sep = ""))
sponsors_session_contacts  <- read.csv(paste(tables_subfolder, 'sponsors_session_contacts.csv', sep = ""))
before_session_contacts  <- read.csv(paste(tables_subfolder, 'before_session_contacts.csv', sep = ""))

authors_table <- left_join(authors_table, bills_conditions, by = 'bill_id_new')

event_table$event <- as.integer(event_table$event)
event_table <- event_table %>%
  arrange(bill_id_new, date_number) %>%
  group_by(bill_id_new) %>%
  mutate(
    chamber_id_filled = ifelse(is.na(chamber_id) & date_number == min(date_number, na.rm = TRUE),
                               chamber_1,
                               zoo::na.locf(chamber_id, na.rm = FALSE)),
    committee_id_filled = ifelse(event == 9 & is.na(committee_id_event), zoo::na.locf(committee_id_event), committee_id_event)
  ) %>%
  ungroup()

# Create event_new = event
event_table$event_new <- event_table$event

# When event == 9, chamber_id_filled == chamber_1, and if committee_id_event == committee_chamber_1_type_1, save event_new = 91
event_table$event_new[(event_table$event == 9) & (event_table$chamber_id_filled == event_table$chamber_1) & (event_table$committee_id_filled == event_table$committee_chamber_1_type_1)] <- 91

# When event == 9, chamber_id_filled == chamber_1 and if committee_id_event != committee_chamber_1_type_1, event_new = 92
event_table$event_new[(event_table$event == 9) & (event_table$chamber_id_filled == event_table$chamber_1) & (event_table$committee_id_filled != event_table$committee_chamber_1_type_1)] <- 92

# When event == 9, chamber_id_filled == chamber_2, and if committee_id_event == committee_chamber_2_type_1, save event_new = 91
event_table$event_new[(event_table$event == 9) & (event_table$chamber_id_filled != event_table$chamber_1) & (event_table$committee_id_filled == event_table$committee_chamber_2_type_1)] <- 91

# When event == 9, chamber_id_filled == chamber_2 and if committee_id_event != committee_chamber_2_type_1, event_new = 92
event_table$event_new[(event_table$event == 9) & (event_table$chamber_id_filled != event_table$chamber_1) & (event_table$committee_id_filled != event_table$committee_chamber_2_type_1)] <- 92

event_table <- event_table %>%
  mutate(chamber_event_1 = if_else(chamber_id_filled == chamber_1, TRUE, FALSE))

min_date_numbers <- event_table %>%
  group_by(bill_id_new) %>%
  summarise(min_date_number = min(date_number, na.rm = TRUE))

# Merge the minimum date numbers back into the original dataframe
event_table <- left_join(event_table, min_date_numbers, by = 'bill_id_new')
event_table <- event_table[complete.cases(event_table$chamber_event_1), ]
# Calculate 'length_since_introduction'
event_table <- event_table %>%
  mutate(length_since_introduction = date_number - min_date_number)


merged_df <- event_table %>%
  group_by(bill_id_new) %>%
  summarize(largest_length_since_introduction = max(length_since_introduction)) %>%
  ungroup()

event_table_processed <- left_join(event_table, merged_df, by = "bill_id_new") %>%
  distinct(bill_id_new, .keep_all = TRUE) %>%
  left_join(
    committees_table %>%
      select(bill_id_new, passes_first_referral, party_1_count, party_2_count, committee_size),
    by = "bill_id_new"
  ) %>%
  left_join(
    authors_table %>%
      group_by(bill_id_new) %>%
      summarize(number_main_sponsors = n_distinct(people_id)),
    by = "bill_id_new"
  ) %>%
  left_join(bills_conditions, by = "bill_id_new") %>%
  mutate(
    intro_session_id = session_id,
    intro_chamber = chamber_1,
    is_assembly_bill = chamber_1 == 73,
    partisan_D = party_1_count > 0 & party_2_count == 0,
    partisan_R = party_1_count == 0 & party_2_count > 0,
    bipartisan = party_1_count > 0 & party_2_count > 0,
    number_main_sponsors,
    largest_length_since_introduction = largest_length_since_introduction * 12,
    INTRO = INTRO,
    AIC = AIC,
    ABC = ABC,
    PASS = PASS,
    LAW = LAW,
    is_matched_bill = coalesce(as.logical(is_matched_bill), FALSE)
  ) %>%
  select(
    bill_id_new,
    intro_session_id,
    is_assembly_bill,
    passed,
    passes_first_referral,
    party_1_count,
    party_2_count,
    committee_size,
    number_main_sponsors,
    largest_length_since_introduction,
    partisan_D,
    partisan_R,
    bipartisan,
    INTRO = INTRO,
    AIC = AIC,
    ABC = ABC,
    PASS = PASS,
    LAW = LAW,
    is_matched_bill
  )

bills_summary <- event_table_processed %>%
  mutate(`Is Assembly bill` = as.numeric(is_assembly_bill),
         `Number main sponsors` = number_main_sponsors,
         `Passed first committee` = as.numeric(as.logical(passes_first_referral)),
         `Passed` = as.numeric(as.logical(passed)),
         `Partisan R` = as.numeric(partisan_R),
         `Bipartisan` = as.numeric(bipartisan),
         `Length history bill (mo.)` = largest_length_since_introduction,
         `2019-2020 LS` = as.numeric(intro_session_id == 1644),
         `Action in committee` = AIC,
         `Action beyond committee` = ABC,
         `Passed House of Origin` = PASS,
         `Became Law` = LAW,
         Type = ifelse(is_matched_bill == TRUE, 'At least a lobbying contact', 'No lobbying contact'),
         Type1 = is_matched_bill==TRUE) %>%
  select(`Is Assembly bill`,
         `Number main sponsors`,
         `Partisan R`,
         `Bipartisan`,
         `Length history bill (mo.)`,
         `2019-2020 LS`,
          INTRO,
         `Action in committee`,
         `Action beyond committee`,
         `Passed House of Origin`,
         `Became Law`,
         Type, Type1)
 
datasummary( `Is Assembly bill`+
               `Number main sponsors`+
               #`Partisan R`+
               `Bipartisan`+
               `Length history bill (mo.)` + `2019-2020 LS`+
               `Action in committee`+
             `Action beyond committee`+
             `Passed House of Origin`+
             `Became Law`~Type*(N + Mean + SD), 
             data = bills_summary, output = paste0(latex_tables_subfolder, 'bills_summary.tex'))

relation_matrix <- matrix(0, nrow = 5, ncol = 5, dimnames = list(c("INTRO", "AIC", "ABC", "PASS", "LAW"), c("INTRO", "AIC", "ABC", "PASS", "LAW")))

cols <- c("INTRO", "AIC", "ABC", "PASS", "LAW")
# Calculate the share for each pair of conditions
for (i in cols) {
  for (j in cols) {
    # Calculate the share for (i, j)
    relation_matrix[i, j] <- sum(bills_conditions[, j] == 1 & bills_conditions[, i] == 1) / sum(bills_conditions[, i] == 1)
  }
}

relation_table <- as.data.frame(relation_matrix)

datasummary_df(relation_table, output = paste0(latex_tables_subfolder, 'bills_progress.tex'),rownames = TRUE)

data_frame(relation_matrix)
contact_grouped <- left_join(
  contact_grouped,
  event_table %>% select('date', 'event', 'bill_id_new', 'committee_id_event', 'length_since_introduction', 'chamber_event_1', 'chamber_id', 'chamber_1', 'chamber_2', 'event_new', 'chamber_id_filled', 'passed'),
  by = c('date', 'event', 'bill_id_new')
)

max_sponsors_contacts <- sponsors_contacts_table %>%
  group_by(bill_id_new) %>%
  summarize(
    distinct_lobbyists = max(distinct_lobbyists, na.rm = TRUE),
    distinct_retained_lobbyists = max(distinct_retained_lobbyists, na.rm = TRUE),
    distinct_employed_lobbyists = max(distinct_employed_lobbyists, na.rm = TRUE)
  )


# Assuming 'contact_grouped' is your DataFrame
contact_grouped <- left_join(
  contact_grouped,
  max_sponsors_contacts,
  by = "bill_id_new"
)


contact_grouped_mutated <- contact_grouped %>%
  #filter(distinct_lobbyists>0) %>%
  #filter((distinct_retained_lobbyists == 1) & (distinct_lobbyists == 1)) %>%
  mutate(across(starts_with("sum_") & 
                  !matches("sum_is_chamber_[12]"), 
                ~ . / total_contacts, 
                .names = "share_{col}"))

contact_grouped_mutated$length_since_introduction_floor <- floor(contact_grouped_mutated$length_since_introduction*6)

heatmap_data <- contact_grouped_mutated %>%
  group_by(event_new, chamber_event_1) %>%
  summarise(across(starts_with("share_"), ~ mean(., na.rm = TRUE)))
# Create the heatmap using ggplot2
heatmap_data <- heatmap_data[complete.cases(heatmap_data$chamber_event_1), ]
heatmap_data <- heatmap_data[heatmap_data$event_new != 9, ]

heatmap_data <- heatmap_data %>%
  mutate(
    event_new = factor(
      recode(
        as.character(event_new),
        `1` = "Introduced",
        `91` = "Refer committee 1",
        `10` = "Passed committee",
        `92` = "Refer committee(s) 2+",
        #`2` = "Engrossed",
        `4` = "Passed",
        `6` = "Failed",
        #`7` = "Override",
        `5` = "Vetoed",
        `8` = "Chaptered",
        #`3` = "Enrolled",
        # Add more recodes for other values as needed
      ),
      levels = c("Introduced", "Refer committee 1", "Passed committee", "Refer committee(s) 2+",
                 "Passed", "Failed", "Vetoed", "Chaptered")
    )
  )


heatmap_data <- heatmap_data %>%
  mutate(chamber_event_1_string = if_else(chamber_event_1 == TRUE, "Chamber Of Origin", "Second Chamber"))

heatmap_data_long <- heatmap_data[!is.na(heatmap_data$event_new),] %>%
  pivot_longer(cols = c(
    "share_sum_sponsors_chamber_1_type_1",
    "share_sum_sponsors_chamber_1_type_2",
    "share_sum_sponsors_chamber_2_type_1",
    "share_sum_sponsors_chamber_2_type_2",
    #"share_sum_committee_members_chamber_1_type_1",
    #"share_sum_committee_members_chamber_1_type_2",
    #"share_sum_committee_members_chamber_2_type_1",
    #"share_sum_committee_members_chamber_2_type_2",
    "share_sum_committee_chamber_1_type_1",
    "share_sum_committee_chamber_1_type_2",
    "share_sum_committee_chamber_2_type_1",
    "share_sum_committee_chamber_2_type_2",
    "share_sum_committee_members_chamber_1_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_1_type_2_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_2_excluding_sponsors",
    "share_sum_member_chamber_1_excluding_sponsors_committee_members",
    "share_sum_member_chamber_2_excluding_sponsors_committee_members"
    #"share_sum_member_chamber_1",
    #"share_sum_member_chamber_2"
  ), names_to = "variable", values_to = "value")

heatmap_data_long <- heatmap_data_long %>%
  mutate(
    variable_name = case_when(
      variable == "share_sum_sponsors_chamber_1_type_1" ~ "Main sponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_1_type_2" ~ "Cosponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_2_type_1" ~ "Main Sponsors in chamber 2",
      variable == "share_sum_sponsors_chamber_2_type_2" ~ "Cosponsors in chamber 2",
      variable == "share_sum_committee_members_chamber_1_type_1" ~ "Committee members in chamber 1 (committee 1)",
      variable == "share_sum_committee_members_chamber_1_type_2" ~ "Committee members in chamber 1 (committee 2+)",
      variable == "share_sum_committee_members_chamber_2_type_1" ~ "Committee members in chamber 2 (committee 1)",
      variable == "share_sum_committee_members_chamber_2_type_2" ~ "Committee members in chamber 2 (committee 2+)",
      variable == "share_sum_committee_members_chamber_1_type_1_excluding_sponsors" ~ "Committee members in chamber 1 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_1_type_2_excluding_sponsors" ~ "Committee members in chamber 1 (committee 2+, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_1_excluding_sponsors" ~ "Committee members in chamber 2 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_2_excluding_sponsors" ~ "Committee members in chamber 2 (excl. sponsors)",
      variable == "share_sum_member_chamber_1_excluding_sponsors_committee_members" ~ "Members in chamber 1 (excl. sponsors, comm. members)",
      variable == "share_sum_member_chamber_2_excluding_sponsors_committee_members" ~ "Members in chamber 2 (excl. sponsors, comm. members)",
      variable == "share_sum_committee_chamber_1_type_1" ~ "Committee in chamber 1 (committee 1)",
      variable == "share_sum_committee_chamber_1_type_2" ~ "Committee in chamber 1 (committee 2+)",
      variable == "share_sum_committee_chamber_2_type_1" ~ "Committee in chamber 2 (committee 1)",
      variable == "share_sum_committee_chamber_2_type_2" ~ "Committee in chamber 2 (committee 2+)",
      TRUE ~ as.character(variable)  # Keep the original name for other variables
    )
  )
heatmap_plot <- ggplot(heatmap_data_long, aes(x = variable_name, y = interaction(event_new, chamber_event_1_string), fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +  # Adjust color scale as needed
  labs(title = "Shares of direct contacts across the progress of bills",
       y = "Bill event and chamber",
       x = "Contact target",
       fill = "Share") +
  theme_minimal() + # You can customize the theme as needed
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Adjust angle and hjust as needed
    axis.text.y = element_text(angle = 0, hjust = 1)  # Adjust angle and hjust as needed
  )
ggsave(paste0(figures_subfolder, "contacts_heatmap.png"), heatmap_plot, width = 12, height = 8, units = "in", dpi = 300)


contact_grouped_mutated <- contact_grouped %>%
  filter(distinct_lobbyists>0) %>%
  filter((distinct_retained_lobbyists == 1) & (distinct_lobbyists == 1)) %>%
  mutate(across(starts_with("sum_") & 
                  !matches("sum_is_chamber_[12]"), 
                ~ . / total_contacts, 
                .names = "share_{col}"))

contact_grouped_mutated$length_since_introduction_floor <- floor(contact_grouped_mutated$length_since_introduction*6)

heatmap_data <- contact_grouped_mutated %>%
  group_by(event_new, chamber_event_1) %>%
  summarise(across(starts_with("share_"), ~ mean(., na.rm = TRUE)))
# Create the heatmap using ggplot2
heatmap_data <- heatmap_data[complete.cases(heatmap_data$chamber_event_1), ]
heatmap_data <- heatmap_data[heatmap_data$event_new != 9, ]

heatmap_data <- heatmap_data %>%
  mutate(
    event_new = recode(
      event_new,
      `1` = "Introduced",
      `2` = "Engrossed",
      `3` = "Enrolled",
      `4` = "Passed",
      `5` = "Vetoed",
      `6` = "Failed",
      `7` = "Override",
      `8` = "Chaptered",
      `10` = "Passed committee",
      `91` = "Refer committee 1",
      `92` = "Refer committee(s) 2+",
      # Add more recodes for other values as needed
    ),
  )

heatmap_data <- heatmap_data %>%
  mutate(chamber_event_1_string = if_else(chamber_event_1 == TRUE, "Chamber 1", "Chamber 2"))

heatmap_data_long <- heatmap_data %>%
  pivot_longer(cols = c(
    "share_sum_sponsors_chamber_1_type_1",
    "share_sum_sponsors_chamber_1_type_2",
    "share_sum_sponsors_chamber_2_type_1",
    "share_sum_sponsors_chamber_2_type_2",
    #"share_sum_committee_members_chamber_1_type_1",
    #"share_sum_committee_members_chamber_1_type_2",
    #"share_sum_committee_members_chamber_2_type_1",
    #"share_sum_committee_members_chamber_2_type_2",
    "share_sum_committee_chamber_1_type_1",
    "share_sum_committee_chamber_1_type_2",
    "share_sum_committee_chamber_2_type_1",
    "share_sum_committee_chamber_2_type_2",
    "share_sum_committee_members_chamber_1_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_1_type_2_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_2_excluding_sponsors",
    "share_sum_member_chamber_1_excluding_sponsors_committee_members",
    "share_sum_member_chamber_2_excluding_sponsors_committee_members"
    #"share_sum_member_chamber_1",
    #"share_sum_member_chamber_2"
  ), names_to = "variable", values_to = "value")

heatmap_data_long <- heatmap_data_long %>%
  mutate(
    variable_name = case_when(
      variable == "share_sum_sponsors_chamber_1_type_1" ~ "Main sponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_1_type_2" ~ "Cosponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_2_type_1" ~ "Main Sponsors in chamber 2",
      variable == "share_sum_sponsors_chamber_2_type_2" ~ "Cosponsors in chamber 2",
      variable == "share_sum_committee_members_chamber_1_type_1" ~ "Committee members in chamber 1 (committee 1)",
      variable == "share_sum_committee_members_chamber_1_type_2" ~ "Committee members in chamber 1 (committee 2+)",
      variable == "share_sum_committee_members_chamber_2_type_1" ~ "Committee members in chamber 2 (committee 1)",
      variable == "share_sum_committee_members_chamber_2_type_2" ~ "Committee members in chamber 2 (committee 2+)",
      variable == "share_sum_committee_members_chamber_1_type_1_excluding_sponsors" ~ "Committee members in chamber 1 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_1_type_2_excluding_sponsors" ~ "Committee members in chamber 1 (committee 2+, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_1_excluding_sponsors" ~ "Committee members in chamber 2 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_2_excluding_sponsors" ~ "Committee members in chamber 2 (excl. sponsors)",
      variable == "share_sum_member_chamber_1_excluding_sponsors_committee_members" ~ "Members in chamber 1 (excl. sponsors, comm. members)",
      variable == "share_sum_member_chamber_2_excluding_sponsors_committee_members" ~ "Members in chamber 2 (excl. sponsors, comm. members)",
      variable == "share_sum_committee_chamber_1_type_1" ~ "Committee in chamber 1 (committee 1)",
      variable == "share_sum_committee_chamber_1_type_2" ~ "Committee in chamber 1 (committee 2+)",
      variable == "share_sum_committee_chamber_2_type_1" ~ "Committee in chamber 2 (committee 1)",
      variable == "share_sum_committee_chamber_2_type_2" ~ "Committee in chamber 2 (committee 2+)",
      TRUE ~ as.character(variable)  # Keep the original name for other variables
    )
  )

heatmap_plot <- ggplot(heatmap_data_long, aes(x = variable_name, y = interaction(event_new, chamber_event_1_string), fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +  # Adjust color scale as needed
  labs(title = "Shares of direct contacts across the progress of bills (with 1 retained lobbyist)",
       y = "Bill event and chamber",
       x = "Contact target",
       fill = "Share") +
  theme_minimal() + # You can customize the theme as needed
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Adjust angle and hjust as needed
    axis.text.y = element_text(angle = 0, hjust = 1)  # Adjust angle and hjust as needed
  )
ggsave(paste0(figures_subfolder, "contacts_heatmap_filtered_1.png"), heatmap_plot, width = 12, height = 8, units = "in", dpi = 300)

contact_grouped_mutated <- contact_grouped %>%
  filter(distinct_lobbyists>3) %>%
  #filter((distinct_retained_lobbyists == 1) & (distinct_lobbyists == 1)) %>%
  mutate(across(starts_with("sum_") & 
                  !matches("sum_is_chamber_[12]"), 
                ~ . / total_contacts, 
                .names = "share_{col}"))

contact_grouped_mutated$length_since_introduction_floor <- floor(contact_grouped_mutated$length_since_introduction*6)

heatmap_data <- contact_grouped_mutated %>%
  group_by(event_new, chamber_event_1) %>%
  summarise(across(starts_with("share_"), ~ mean(., na.rm = TRUE)))
# Create the heatmap using ggplot2
heatmap_data <- heatmap_data[complete.cases(heatmap_data$chamber_event_1), ]
heatmap_data <- heatmap_data[heatmap_data$event_new != 9, ]

heatmap_data <- heatmap_data %>%
  mutate(
    event_new = recode(
      event_new,
      `1` = "Introduced",
      `2` = "Engrossed",
      `3` = "Enrolled",
      `4` = "Passed",
      `5` = "Vetoed",
      `6` = "Failed",
      `7` = "Override",
      `8` = "Chaptered",
      `10` = "Passed committee",
      `91` = "Refer committee 1",
      `92` = "Refer committee(s) 2+",
      # Add more recodes for other values as needed
    ),
  )

heatmap_data <- heatmap_data %>%
  mutate(chamber_event_1_string = if_else(chamber_event_1 == TRUE, "Chamber 1", "Chamber 2"))

heatmap_data_long <- heatmap_data %>%
  pivot_longer(cols = c(
    "share_sum_sponsors_chamber_1_type_1",
    "share_sum_sponsors_chamber_1_type_2",
    "share_sum_sponsors_chamber_2_type_1",
    "share_sum_sponsors_chamber_2_type_2",
    #"share_sum_committee_members_chamber_1_type_1",
    #"share_sum_committee_members_chamber_1_type_2",
    #"share_sum_committee_members_chamber_2_type_1",
    #"share_sum_committee_members_chamber_2_type_2",
    "share_sum_committee_chamber_1_type_1",
    "share_sum_committee_chamber_1_type_2",
    "share_sum_committee_chamber_2_type_1",
    "share_sum_committee_chamber_2_type_2",
    "share_sum_committee_members_chamber_1_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_1_type_2_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_1_excluding_sponsors",
    "share_sum_committee_members_chamber_2_type_2_excluding_sponsors",
    "share_sum_member_chamber_1_excluding_sponsors_committee_members",
    "share_sum_member_chamber_2_excluding_sponsors_committee_members"
    #"share_sum_member_chamber_1",
    #"share_sum_member_chamber_2"
  ), names_to = "variable", values_to = "value")

heatmap_data_long <- heatmap_data_long %>%
  mutate(
    variable_name = case_when(
      variable == "share_sum_sponsors_chamber_1_type_1" ~ "Main sponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_1_type_2" ~ "Cosponsors in chamber 1",
      variable == "share_sum_sponsors_chamber_2_type_1" ~ "Main Sponsors in chamber 2",
      variable == "share_sum_sponsors_chamber_2_type_2" ~ "Cosponsors in chamber 2",
      variable == "share_sum_committee_members_chamber_1_type_1" ~ "Committee members in chamber 1 (committee 1)",
      variable == "share_sum_committee_members_chamber_1_type_2" ~ "Committee members in chamber 1 (committee 2+)",
      variable == "share_sum_committee_members_chamber_2_type_1" ~ "Committee members in chamber 2 (committee 1)",
      variable == "share_sum_committee_members_chamber_2_type_2" ~ "Committee members in chamber 2 (committee 2+)",
      variable == "share_sum_committee_members_chamber_1_type_1_excluding_sponsors" ~ "Committee members in chamber 1 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_1_type_2_excluding_sponsors" ~ "Committee members in chamber 1 (committee 2+, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_1_excluding_sponsors" ~ "Committee members in chamber 2 (committee 1, excl. sponsors)",
      variable == "share_sum_committee_members_chamber_2_type_2_excluding_sponsors" ~ "Committee members in chamber 2 (excl. sponsors)",
      variable == "share_sum_member_chamber_1_excluding_sponsors_committee_members" ~ "Members in chamber 1 (excl. sponsors, comm. members)",
      variable == "share_sum_member_chamber_2_excluding_sponsors_committee_members" ~ "Members in chamber 2 (excl. sponsors, comm. members)",
      variable == "share_sum_committee_chamber_1_type_1" ~ "Committee in chamber 1 (committee 1)",
      variable == "share_sum_committee_chamber_1_type_2" ~ "Committee in chamber 1 (committee 2+)",
      variable == "share_sum_committee_chamber_2_type_1" ~ "Committee in chamber 2 (committee 1)",
      variable == "share_sum_committee_chamber_2_type_2" ~ "Committee in chamber 2 (committee 2+)",
      TRUE ~ as.character(variable)  # Keep the original name for other variables
    )
  )

heatmap_plot <- ggplot(heatmap_data_long, aes(x = variable_name, y = interaction(event_new, chamber_event_1_string), fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +  # Adjust color scale as needed
  labs(title = "Shares of direct contacts across the progress of bills (with at least 4 lobbyists)",
       y = "Bill event and chamber",
       x = "Contact target",
       fill = "Share") +
  theme_minimal() + # You can customize the theme as needed
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Adjust angle and hjust as needed
    axis.text.y = element_text(angle = 0, hjust = 1)  # Adjust angle and hjust as needed
  )
ggsave(paste0(figures_subfolder, "contacts_heatmap_filtered_many.png"), heatmap_plot, width = 12, height = 8, units = "in", dpi = 300)


# Create an empty list to store the plots
plots <- list()

# Specify the directory where you want to save the plots
titles <- c("Distinct lobbyists (bills with 1 lobbyist)", 
            "Distinct lobbyists (bills with 2 lobbyists)", 
            "Distinct lobbyists (bills with 3 lobbyists)", 
            "Distinct lobbyists (bills with 4 lobbyists)")
# Loop through the dataframes and create a plot for each
for(i in 1:4) {
  
  title <-  paste("Distinct lobbyists (bills with ", i, " principal lobbyist)", sep = "")
  df <- sponsors_contacts_table[sponsors_contacts_table$principal_lobbyist == i,] %>%
    group_by(event_bimonth_sequence) %>%
    summarise(
      avg_distinct_lobbyists = mean(distinct_lobbyists),
      avg_distinct_retained_lobbyists = mean(distinct_retained_lobbyists),
      avg_distinct_employed_lobbyists = mean(distinct_employed_lobbyists),
      sd_distinct_lobbyists = sd(distinct_lobbyists),
      sd_distinct_retained_lobbyists = sd(distinct_retained_lobbyists),
      sd_distinct_employed_lobbyists = sd(distinct_employed_lobbyists),
    )
  
  p <- ggplot(df, aes(x = event_bimonth_sequence)) +
    geom_line(aes(y = avg_distinct_lobbyists), color = "red") +
    geom_ribbon(aes(ymin = avg_distinct_lobbyists - sd_distinct_lobbyists, ymax = avg_distinct_lobbyists + sd_distinct_lobbyists, fill = "All lobbyists"), alpha = 0.2) +
    geom_line(aes(y = avg_distinct_retained_lobbyists), color = "blue") +
    geom_ribbon(aes(ymin = avg_distinct_retained_lobbyists - sd_distinct_retained_lobbyists, ymax = avg_distinct_retained_lobbyists + sd_distinct_retained_lobbyists, fill = "Retained lobbyists"), alpha = 0.2) +
    geom_line(aes(y = avg_distinct_employed_lobbyists), color = "green") +
    geom_ribbon(aes(ymin = avg_distinct_employed_lobbyists - sd_distinct_employed_lobbyists, ymax = avg_distinct_employed_lobbyists + sd_distinct_employed_lobbyists, fill = "Employed lobbyists"), alpha = 0.2) +
    labs(title = titles[i], x = "Bi-month period after introduction of the bill", y = "Distinct lobbyists") +
    theme_minimal() +
    scale_fill_manual(values = c("All lobbyists" = "red", "Retained lobbyists" = "blue", "Employed lobbyists" = "green")) +
    theme(legend.title = element_blank()) +
    scale_x_continuous(breaks = unique(df$event_bimonth_sequence))
  plots[[i]] <- p
  
  # Save the plot
  ggsave(paste0(figures_subfolder, "sponsors_contacts_", i, ".png"), plot = p, width = 10, height = 5)
}
plots <- list()

# Loop through the different total_lobbyists values
for (i in 1:4) {
  
  # Filter data for the specific total_lobbyists value
  df_subset <- contacts_history[contacts_history$total_lobbyists == i, ]
  
  # Create the plot
  if (i == 1){
    p <- ggplot(df_subset, aes(x = event_bimonth_sequence)) +
      geom_line(aes(y = contacts_1), color = "red") +
      labs(title = paste("Contacts with main sponsors (bills with", i, "principal lobbyists)"),
           x = "Month period after introduction of the bill", y = "Number of bills") +
      theme_minimal() +
      scale_x_continuous(breaks = seq(min(df_subset$event_bimonth_sequence), max(df_subset$event_bimonth_sequence*2), 1))
    
    plots[[i]] <- p
  }
  else if (i == 2){
  p <- ggplot(df_subset, aes(x = event_bimonth_sequence)) +
    geom_line(aes(y = contacts_1), color = "red") +
    geom_line(aes(y = contacts_2), color = "blue") +
    labs(title = paste("Contacts with main sponsors (bills with", i, "principal lobbyists)"),
         x = "Month period after introduction of the bill", y = "Number of bills") +
    theme_minimal() +
    scale_x_continuous(breaks = seq(min(df_subset$event_bimonth_sequence), max(df_subset$event_bimonth_sequence*2), 1))
  
  plots[[i]] <- p
  }
  else if (i == 3){
   p <- ggplot(df_subset, aes(x = event_bimonth_sequence)) +
    geom_line(aes(y = contacts_1), color = "red") +
    geom_line(aes(y = contacts_2), color = "blue") +
    geom_line(aes(y = contacts_3), color = "green") +
   labs(title = paste("Contacts with main sponsors (bills with", i, "principal lobbyists)"),
        x = "Month period after introduction of the bill", y = "Number of bills") +
     theme_minimal() +
     scale_x_continuous(breaks = seq(min(df_subset$event_bimonth_sequence), max(df_subset$event_bimonth_sequence*2), 1))
   
   plots[[i]] <- p
  }
  else if (i == 4){
    p <- ggplot(df_subset, aes(x = event_bimonth_sequence)) +
      geom_line(aes(y = contacts_1), color = "red") +
      geom_line(aes(y = contacts_2), color = "blue") +
      geom_line(aes(y = contacts_3), color = "green") +
      geom_line(aes(y = contacts_4), color = "orange") +
    labs(title = paste("Contacts with main sponsors (bills with", i, "principal lobbyists)"),
         x = "Month period after introduction of the bill", y = "Number of bills") +
    theme_minimal() +
    scale_x_continuous(breaks = seq(min(df_subset$event_bimonth_sequence), max(df_subset$event_bimonth_sequence*2), 1))
    
    plots[[i]] <- p
  }
  # Add the plot to the list
  
  # Save the plot
  ggsave(paste0(figures_subfolder, "contacts_plot_sum_", i, ".png"), plot = p, width = 10, height = 5)
}

plots <- list()
# Loop through the different total_lobbyists values
for (i in 1:4) {
  
  # Filter data for the specific total_lobbyists value
  df_subset <- contacts_history_avg[contacts_history_avg$total_lobbyists == i, ]
  
  # Create the plot
  p <- ggplot(df_subset, aes(x = event_bimonth_sequence)) +
    geom_line(aes(y = avg_contacts_1), color = "red") +
    geom_line(aes(y = avg_contacts_2), color = "blue") +
    geom_line(aes(y = avg_contacts_3), color = "green") +
    geom_line(aes(y = avg_contacts_4), color = "orange") +
    labs(title = paste("Shares of contacts with main sponsors (bills with", i, "principal lobbyists)"),
         x = "Bi-month period after introduction of the bill", y = "Shares of contacts with main sponsors") +
    theme_minimal() +
    scale_x_continuous(breaks = seq(min(df_subset$event_bimonth_sequence), max(df_subset$event_bimonth_sequence), 1))
  
  # Add the plot to the list
  plots[[i]] <- p
  
  # Save the plot
  ggsave(paste0(figures_subfolder, "contacts_plot_avg_", i, ".png"), plot = p, width = 10, height = 5)
}


authors_table$is_matched_bill <- as.logical(authors_table$is_matched_bill)
authors_table$is_bill_lobbyist_lawmaker_matched <- as.logical(authors_table$is_bill_lobbyist_lawmaker_matched)
authors_table$passed <- as.logical(authors_table$passed)
authors_table$matched_not_authored <- authors_table$is_matched_bill & !authors_table$is_bill_lobbyist_lawmaker_matched

bill_weights <- authors_table %>%
  group_by(bill_id_new) %>%
  summarise(weight = 1 / n_distinct(people_id))
# result <- authors_table %>%
#   group_by(people_id, session_id, party_id, role_id) %>%
#   summarise(
#     total_bills = n(),
#     lobbied_bills = sum(is_matched_bill),
#     lobbied_passed_bills = sum(is_matched_bill & passed),
#     authored_bills = sum(is_bill_lobbyist_lawmaker_matched),
#     authored_passed_bills = sum(is_bill_lobbyist_lawmaker_matched & passed),
#   )

authors_table$INTRO <- as.logical(authors_table$INTRO)
authors_table$AIC <- as.logical(authors_table$AIC)
authors_table$ABC <- as.logical(authors_table$ABC)
authors_table$PASS <- as.logical(authors_table$PASS)
authors_table$LAW <- as.logical(authors_table$LAW)
authors_summary <- authors_table %>%
  left_join(bill_weights, by = "bill_id_new") %>%
  group_by(people_id, session_id, party_id, role_id) %>%
  summarise(
    LES = mean(LES),
    LPS = mean(LPS),
    total_INTRO = sum(INTRO),
    total_AIC = sum(AIC),
    total_ABC = sum(ABC),
    total_PASS = sum(PASS),
    total_LAW = sum(LAW),
    authored_INTRO = sum(is_bill_lobbyist_lawmaker_matched & INTRO),
    authored_AIC = sum(is_bill_lobbyist_lawmaker_matched & AIC),
    authored_ABC = sum(is_bill_lobbyist_lawmaker_matched & ABC),
    authored_PASS = sum(is_bill_lobbyist_lawmaker_matched & PASS),
    authored_LAW = sum(is_bill_lobbyist_lawmaker_matched & LAW),
    ratio_INTRO = authored_INTRO/total_INTRO,
    ratio_AIC = authored_AIC/total_AIC,
    ratio_ABC = authored_ABC/total_ABC,
    ratio_PASS = authored_PASS/total_PASS,
    ratio_LAW = authored_LAW/total_LAW,
    total_bills = n(),
    passed_bills = sum(passed),
    lobbied_bills = sum(is_matched_bill),
    lobbied_passed_bills = sum(is_matched_bill & passed),
    authored_bills = sum(is_bill_lobbyist_lawmaker_matched),
    authored_passed_bills = sum(is_bill_lobbyist_lawmaker_matched & passed),
    lobbied_not_authored_bills = sum(matched_not_authored),
    lobbied_not_authored_passed_bills = sum(matched_not_authored & passed),
    ratio_passed_total = passed_bills/total_bills,
    ratio_authored_total = authored_bills/total_bills,
    ratio_passed_authored = authored_passed_bills/authored_bills,
    weighted_total_bills = sum(weight),
    weighted_passed_bills = sum(weight*passed),
    weighted_lobbied_bills = sum(weight*is_matched_bill),
    weighted_lobbied_passed_bills = sum(weight*(is_matched_bill & passed)),
    weighted_lobbied_not_authored_bills = sum(weight*matched_not_authored),
    weighted_lobbied_not_authored_passed_bills = sum(weight*(matched_not_authored & passed)),
    weighted_authored_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched)),
    weighted_authored_passed_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched & passed)),
    weighted_ratio_passed_lobbied = weighted_lobbied_passed_bills/weighted_lobbied_bills,
    weighted_ratio_passed_lobbied_not_authored = weighted_lobbied_not_authored_passed_bills/weighted_lobbied_not_authored_bills,
    weighted_ratio_passed_total = weighted_passed_bills/weighted_total_bills,
    weighted_ratio_authored_total = weighted_authored_bills/weighted_total_bills,
    weighted_ratio_passed_authored = weighted_authored_passed_bills/weighted_authored_bills,
    weighted_ratio_authored_passed = weighted_authored_passed_bills/weighted_passed_bills,
  )# Create a new window with two side-by-side plots
# Set smaller margins

gg <- ggplot(authors_summary, aes(x = ratio_INTRO)) +
  geom_point(aes(y = ratio_LAW, color = interaction(party_id, role_id))) +
  scale_color_manual(values = c("blue", 'red',"cyan", "pink"), 
                     name = "Chamber/Party", 
                     labels = c("Rep (D)", "Rep (R)", "Sen (D)", "Sen (R)"))  +
  labs(title = "", x = "Ratio passed/introduced (total bills)", y = "Ratio passed/introduced (bills with previous lobbying input)") +
  theme_minimal()

# Save the ggplot image
ggsave(paste0(figures_subfolder, "ratio_bills.png"), gg, width = 8, height = 5, units = "in", dpi = 300)

hist_ggplot <- ggplot(authors_summary, aes(x = total_INTRO)) +
  geom_histogram(aes(fill = "Total bills"), binwidth = 5, color = "black", alpha = 0.5) +
  geom_histogram(aes(x = authored_INTRO, fill = "Bills with lobbying input before introduction"), binwidth = 5, color = "black", alpha = 0.5) +
  labs(x = "Count", y = "Frequency") +
  scale_fill_manual(name = "Legend", values = c("green", "purple")) +
  theme_minimal()
ggsave(paste0(figures_subfolder, "histogram_bills.png"),hist_ggplot, width = 8, height = 5, units = "in", dpi = 300)



# Density plot for total_INTRO and authored_INTRO
plot(density(authors_summary$total_INTRO), col = "blue", main = "Density Plot of total_INTRO and authored_INTRO", xlab = "Count")
lines(density(authors_summary$authored_INTRO), col = "red")

# Add legend
legend("topright", legend = c("total_INTRO", "authored_INTRO"), fill = c("blue", "red"))


# Save the plot
ggsave(paste0(figures_subfolder, ""), width = 8, height = 6, units = "in", dpi = 300)

authors_summary <- authors_table %>%
  left_join(bill_weights, by = "bill_id_new") %>%
  group_by(people_id, session_id, party_id, role_id) %>%
  summarise(
    LES = mean(LES),
    LPS = mean(LPS),
    total_INTRO = sum(INTRO),
    total_AIC = sum(AIC),
    total_ABC = sum(ABC),
    total_PASS = sum(PASS),
    total_LAW = sum(LAW),
    authored_INTRO = sum(is_bill_lobbyist_lawmaker_matched & INTRO),
    authored_AIC = sum(is_bill_lobbyist_lawmaker_matched & AIC),
    authored_ABC = sum(is_bill_lobbyist_lawmaker_matched & ABC),
    authored_PASS = sum(is_bill_lobbyist_lawmaker_matched & PASS),
    authored_LAW = sum(is_bill_lobbyist_lawmaker_matched & LAW),
    ratio_INTRO = authored_INTRO/total_INTRO,
    ratio_AIC = authored_AIC/total_AIC,
    ratio_ABC = authored_ABC/total_ABC,
    ratio_PASS = authored_PASS/total_PASS,
    ratio_LAW = authored_LAW/total_LAW,
    total_bills = n(),
    passed_bills = sum(passed),
    lobbied_bills = sum(is_matched_bill),
    lobbied_passed_bills = sum(is_matched_bill & passed),
    authored_bills = sum(is_bill_lobbyist_lawmaker_matched),
    authored_passed_bills = sum(is_bill_lobbyist_lawmaker_matched & passed),
    lobbied_not_authored_bills = sum(matched_not_authored),
    lobbied_not_authored_passed_bills = sum(matched_not_authored & passed),
    ratio_passed_total = passed_bills/total_bills,
    ratio_authored_total = authored_bills/total_bills,
    ratio_passed_authored = authored_passed_bills/authored_bills,
    weighted_total_bills = sum(weight),
    weighted_passed_bills = sum(weight*passed),
    weighted_lobbied_bills = sum(weight*is_matched_bill),
    weighted_lobbied_passed_bills = sum(weight*(is_matched_bill & passed)),
    weighted_lobbied_not_authored_bills = sum(weight*matched_not_authored),
    weighted_lobbied_not_authored_passed_bills = sum(weight*(matched_not_authored & passed)),
    weighted_authored_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched)),
    weighted_authored_passed_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched & passed)),
    weighted_ratio_passed_lobbied = weighted_lobbied_passed_bills/weighted_lobbied_bills,
    weighted_ratio_passed_lobbied_not_authored = weighted_lobbied_not_authored_passed_bills/weighted_lobbied_not_authored_bills,
    weighted_ratio_passed_total = weighted_passed_bills/weighted_total_bills,
    weighted_ratio_authored_total = weighted_authored_bills/weighted_total_bills,
    weighted_ratio_passed_authored = weighted_authored_passed_bills/weighted_authored_bills,
    weighted_ratio_authored_passed = weighted_authored_passed_bills/weighted_passed_bills,
  )# Create a new window with two side-by-side plots
# Set smaller margins

committee_members <- committee_members %>%
  mutate(
    chair = !is.na(chair),
    vice_chair = !is.na(vice_chair)
  )
committee_people <- committee_members %>%
  rename(people_id = people_id_inferred) %>%
  group_by(people_id, session_id) %>%
  summarize(
    distinct_committee_count = n_distinct(committee_id),
    chair_count = sum(chair),
    is_chair = chair_count > 0
  ) %>%
  ungroup()

people_arranged <- people_all %>%
  arrange(people_id, session_id) %>%
  group_by(people_id) %>%
  mutate(
    terms = row_number(),
    freshman = as.logical(terms==1)
  ) %>%
  group_by(people_id, role_id) %>%
  mutate(
    terms_role = row_number()
  ) %>%
  ungroup()

authors_summary$role_id <- factor(authors_summary$role_id)
authors_summary$party_id <- factor(authors_summary$party_id)
# Filter data based on role_id
df_role_1 <- authors_summary[authors_summary$role_id == 1,]
df_role_2 <- authors_summary[authors_summary$role_id == 2,]

authors_summary_temp = authors_summary#[authors_summary$session_id==1813,]
# Create scatter plots for role_id == 1
ggplot(authors_summary_temp, aes(x = weighted_total_bills)) +
  geom_point(aes(y = weighted_ratio_passed_total, color = interaction(party_id, role_id))) +
  scale_color_manual(values = c("blue", 'red',"cyan", "pink"), 
                     name = "Chamber/Party", 
                     labels = c("Rep (D)", "Rep (R)", "Sen (D)", "Sen (R)"))  +
  labs(title = "Effectiveness of lawmakers in each session", x = "Number of bills introduced by lawmaker in session", y = "Ratio passed/introduced bills") +
  theme_minimal()

# Save the plot
ggsave(paste0(figures_subfolder, "effectiveness_lawmakers.png"), width = 8, height = 6, units = "in", dpi = 300)



# Create scatter plots for role_id == 1
ggplot(authors_summary_temp, aes(x = weighted_total_bills)) +
  geom_point(aes(y = weighted_authored_bills, color = interaction(party_id, role_id))) +
  scale_color_manual(values = c("blue", 'red',"cyan", "pink"), 
  name = "Chamber/Party", 
  labels = c("Rep (D)", "Rep (R)", "Sen (D)", "Sen (R)"))  +
  labs(title = "Number of lobbied bills introduced lawmakers in each session", x = "Number of bills introduced by lawmaker in session", y = "Number of bills lobbied") +
  theme_minimal()


# Save the plot
ggsave(paste0(figures_subfolder, "productivity_lawmakers.png"), width = 8, height = 6, units = "in", dpi = 300)

# Create scatter plots for role_id == 1
ggplot(authors_summary_temp, aes(x = weighted_authored_bills)) +
  geom_point(aes(y = weighted_ratio_passed_authored, , color = interaction(party_id, role_id))) +
  scale_color_manual(values = c("blue", 'red',"cyan", "pink"), 
                     name = "Chamber/Party", 
                     labels = c("Rep (D)", "Rep (R)", "Sen (D)", "Sen (R)"))  +
  labs(title = "Effectiveness of lawmakers in each session (only lobbied bills)", x = "Number of bills introduced by lawmaker in session", y = "Ratio passed/introduced lobbied bills") +
  theme_minimal()

ggsave(paste0(figures_subfolder, "effectiveness_lobbying_lawmakers.png"), width = 8, height = 6, units = "in", dpi = 300)



committee_members <- committee_members %>%
  mutate(
    chair = !is.na(chair),
    vice_chair = !is.na(vice_chair)
  )
committee_people <- committee_members %>%
  rename(people_id = people_id_inferred) %>%
  group_by(people_id, session_id) %>%
  summarize(
    distinct_committee_count = n_distinct(committee_id),
    chair_count = sum(chair),
    is_chair = chair_count > 0
  ) %>%
  ungroup()

people_arranged <- people_all %>%
  arrange(people_id, session_id) %>%
  group_by(people_id) %>%
  mutate(
    terms = row_number()
  ) %>%
  group_by(people_id, role_id) %>%
  mutate(
    terms_role = row_number()
  ) %>%
  ungroup()

authors_summary_with_committees <- left_join(authors_summary, committee_people, by = c("people_id", "session_id"))
authors_summary_with_committees <- left_join(authors_summary_with_committees, people_arranged[,c('people_id', 'session_id', 'terms', 'terms_role')], by = c("people_id", "session_id"))

authors_summary_with_committees_all_sessions <- authors_summary_with_committees %>%
  group_by(people_id, party_id, role_id) %>%
  summarise(
    num_sessions = n_distinct(session_id),
    across(where(is.numeric), mean),
    across(where(is.logical), mean)
  )

# Print the summary table
print(authors_summary_with_committees_all_sessions)

result_chambers <- authors_summary_with_committees_all_sessions %>%
  mutate(`No. sessions` = num_sessions,
         `Minority party (R)` = as.numeric(party_id==2),
         `Committee chair` = as.numeric(chair_count),
         `No. committees member` = distinct_committee_count,
         `Minority party (R)` = as.numeric(party_id==2),
         #`Terms` = terms,
         `Bills introduced` = total_INTRO,
         `Bills passed` = total_LAW,
         Chamber = ifelse(role_id == 1, 'Assembly', 'Senate')) %>%
         #Session = ifelse(session_id == 1644, '2019-2020', '2021-2022')) %>%
  select(`No. sessions`,
         `Minority party (R)`,
         `Committee chair`,
         `No. committees member`,
         #`Terms`,
         `Bills introduced`,
         `Bills passed`,
         Chamber
         )

datasummary(Chamber*(`No. sessions`+
                     `Minority party (R)`+
                     `Committee chair`+
                     `No. committees member`+
                     #`Terms`,
                     `Bills introduced`+
                     `Bills passed`) ~ 1*(N + Min + Mean + Median + Max + SD), data = result_chambers, output = paste0(latex_tables_subfolder, 'legislators_summary.tex'))


model <- lm(LPS ~ as.factor(party_id) + is_chair + distinct_committee_count + terms_role, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 2,])
summary(model)


assembly_expenses <- read.csv(paste(tables_subfolder, 'assembly_members_aggregated_expenses.csv', sep = ""))

senate_expenses <- read.csv(paste(tables_subfolder, 'senate_members_aggregated_expenses.csv', sep = ""))

intensity_lobbying  <- read.csv(paste(tables_subfolder, 'intensity_lobbying_before_year_sem.csv', sep = ""))

expenses <- read.csv(paste(tables_subfolder, 'members_aggregated_expenses.csv', sep = ""))
total_bills_by_year <- authors_table %>%
  group_by(intro_year) %>%
  summarise(total_year_bills = n_distinct(bill_id_new))

authors_summary <- authors_table %>%
  left_join(bill_weights, by = "bill_id_new") %>%
  group_by(people_id, party_id, role_id, session_id) %>%
  summarise(
    LES = max(LES),
    LES = max(LPS),
    total_bills = n(),
    passed_bills = sum(passed),
    lobbied_bills = sum(is_matched_bill),
    lobbied_passed_bills = sum(is_matched_bill & passed),
    authored_bills = sum(is_bill_lobbyist_lawmaker_matched),
    authored_passed_bills = sum(is_bill_lobbyist_lawmaker_matched & passed),
    lobbied_not_authored_bills = sum(matched_not_authored),
    lobbied_not_authored_passed_bills = sum(matched_not_authored & passed),
    ratio_passed_total = passed_bills/total_bills,
    ratio_authored_total = authored_bills/total_bills,
    ratio_passed_authored = authored_passed_bills/authored_bills,
    # weighted_total_bills = sum(weight),
    # weighted_passed_bills = sum(weight*passed),
    # weighted_lobbied_bills = sum(weight*is_matched_bill),
    # weighted_lobbied_passed_bills = sum(weight*(is_matched_bill & passed)),
    # weighted_lobbied_not_authored_bills = sum(weight*matched_not_authored),
    # weighted_lobbied_not_authored_passed_bills = sum(weight*(matched_not_authored & passed)),
    # weighted_authored_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched)),
    # weighted_authored_passed_bills = sum(weight*(is_bill_lobbyist_lawmaker_matched & passed)),
    # weighted_ratio_passed_lobbied = weighted_lobbied_passed_bills/weighted_lobbied_bills,
    # weighted_ratio_passed_lobbied_not_authored = weighted_lobbied_not_authored_passed_bills/weighted_lobbied_not_authored_bills,
    # weighted_ratio_passed_total = weighted_passed_bills/weighted_total_bills,
    # weighted_ratio_authored_total = weighted_authored_bills/weighted_total_bills,
    # weighted_ratio_passed_authored = weighted_authored_passed_bills/weighted_authored_bills,
    # weighted_ratio_authored_passed = weighted_authored_passed_bills/weighted_passed_bills,
  ) #%>%
  #left_join(total_bills_by_year, by = "intro_year")

authors_summary$ratio_intro_year <- authors_summary$total_bills/authors_summary$total_year_bills

authors_summary$role_id <- factor(authors_summary$role_id)
authors_summary$party_id <- factor(authors_summary$party_id)
authors_summary$people_id <- as.integer(authors_summary$people_id)
intensity_lobbying$people_id <- as.integer(intensity_lobbying$people_id)

names(authors_summary)[names(authors_summary) == "intro_year"] <- "year"
names(authors_summary)[names(authors_summary) == "intro_semester"] <- "filing_semester"
names(assembly_expenses)[names(assembly_expenses) == "start_year"] <- "year"

authors_summary_with_committees <- left_join(authors_summary, committee_people, by = c("people_id", "session_id"))
authors_summary_with_committees <- left_join(authors_summary_with_committees, people_arranged[,c('people_id', 'session_id', 'terms', 'freshman', 'terms_role')], by = c("people_id", "session_id"))
#authors_summary_with_committees <- left_join(authors_summary_with_committees, intensity_lobbying, by = c("people_id", "session_id"))
#authors_summary_with_committees <- left_join(authors_summary_with_committees, assembly_expenses, by = c("people_id", "session_id"))
authors_summary_with_committees$year <- authors_summary_with_committees$year - 1

authors_summary_with_committees <- left_join(authors_summary_with_committees, intensity_lobbying, by = c("people_id", "year"))
authors_summary_with_committees <- left_join(authors_summary_with_committees, assembly_expenses,  by = c("people_id", "session_id"))

authors_summary_with_committees <- left_join(authors_summary_with_committees, expenses,  by = c("people_id", "session_id"))


people_session_contacts  <- read.csv(paste(tables_subfolder, 'people_session_contacts.csv', sep = ""))
authors_summary_with_committees <- left_join(authors_summary_with_committees,people_session_contacts,  by = c("people_id", "session_id"))

people_session_contacts  <- read.csv(paste(tables_subfolder, 'people_session_contacts_3.csv', sep = ""))

people_session_contacts$is_matched_bill <- as.logical(people_session_contacts$is_matched_bill)
people_session_contacts <- people_session_contacts[people_session_contacts$is_matched_bill!=TRUE,]
authors_summary_with_committees <- left_join(authors_summary_with_committees,people_session_contacts,  by = c("people_id", "session_id"))

authors_summary_with_committees <- left_join(authors_summary_with_committees,sponsors_session_contacts,  by = c("people_id", "session_id"))
authors_summary_with_committees <- left_join(authors_summary_with_committees,before_session_contacts,  by = c("people_id", "session_id"))

cols_to_fill <- c("parties_lobbied_id")  # Add the column names you want to fill
authors_summary_with_committees[cols_to_fill][is.na(authors_summary_with_committees[cols_to_fill])] <- 0
cols_to_fill <- c("is_char")  # Add the column names you want to fill
authors_summary_with_committees[cols_to_fill][is.na(authors_summary_with_committees[cols_to_fill])] <- FALSE
model1 <- lm(log(parties_lobbied_id.x + 1) ~ as.factor(role_id) + as.factor(party_id)  + as.factor(session_id) + is_chair + freshman + terms, data =  authors_summary_with_committees)
model2 <- lm(log(parties_lobbied_id.y + 1) ~ as.factor(role_id) + as.factor(party_id)  + as.factor(session_id) + is_chair + freshman + terms, data =  authors_summary_with_committees)
model3 <- lm(log(parties_lobbied_id + 1) ~ as.factor(role_id) + as.factor(party_id)  + as.factor(session_id) + is_chair + freshman + terms, data =  authors_summary_with_committees)


stargazer(model1,model3, add.lines=list(c('Session fixed effects', 'Yes', 'Yes', 'Yes')), covariate.labels = 
            c("Senator", "Minority party","Chair committee", "Freshman", "Seniority", "(Intercept)"), omit = "session_id", single.row = TRUE, dep.var.labels = c("log-Contacts","log-Contacts (no bill id)"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "small", omit.stat=c("LL","ser","f", "rsq"))

model1 <- lm(log(total_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms.x + I(terms.x^2), data = authors_summary_with_committees)
model2 <- lm(log(total_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms.x + I(terms.x^2) + log(parties_lobbied_id.y+1), data = authors_summary_with_committees)
model3 <- lm(log(total_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + is_chair  + terms.x + as.factor(people_id) + I(terms.x^2)+ as.factor(session_id), data = authors_summary_with_committees)
model4 <- lm(log(total_bills+0.01) ~ as.factor(role_id) +  + as.factor(party_id) + as.factor(session_id) + is_chair + terms.x + I(terms.x^2)+ log(parties_lobbied_id.y+1) + as.factor(people_id), data = authors_summary_with_committees)
model5 <- lm(LES ~ as.factor(role_id) +  + as.factor(party_id) + as.factor(session_id) + is_chair + terms+ I(terms^2)+ log(parties_lobbied_id+1) + as.factor(people_id), data = authors_summary_with_committees)


model4 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id)+ is_chair + freshman + terms.x, data = authors_summary_with_committees)
model5 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair + freshman + terms.x + log(parties_lobbied_id.x+1), data = authors_summary_with_committees)
model6 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair + freshman + terms.x + log(parties_lobbied_id.y+1), data = authors_summary_with_committees)



stargazer(model1,model2,model3,model4, add.lines=list(c('Chamber FE', 'Yes', 'Yes', 'Yes', 'Yes'), c('Session FE', 'Yes', 'Yes', 'Yes', 'Yes'), c('Legislator FE', 'No', 'No', 'Yes', 'Yes')), covariate.labels = 
            c("Minority party", "Chair committee", "Seniority", "Seniority sq.", "Log Lobb. Contacts"), omit = c("session_id", "role_id", "people_id"), single.row = TRUE, dep.var.labels = c("Log sponsored bills", "Log passed bills"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "tiny", omit.stat=c("LL","ser","f", "rsq"))


model1 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms.x + I(terms.x^2), data = authors_summary_with_committees)
model2 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms.x + I(terms.x^2)+ log(parties_lobbied_id.y+1), data = authors_summary_with_committees)
model3 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + is_chair + terms.x + as.factor(people_id) + I(terms.x^2)+ as.factor(session_id), data = authors_summary_with_committees)
model4 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) +  + as.factor(party_id) + as.factor(session_id) + is_chair + terms.x + I(terms.x^2)+ log(parties_lobbied_id.y+1) + as.factor(people_id), data = authors_summary_with_committees)


stargazer(model1,model2,model3,model4, add.lines=list(c('Chamber FE', 'Yes', 'Yes', 'Yes', 'Yes'), c('Session FE', 'Yes', 'Yes', 'Yes', 'Yes'), c('Legislator FE', 'No', 'No', 'Yes', 'Yes')), covariate.labels = 
            c("Minority party", "Chair committee", "Seniority", "Seniority sq.", "Log Lobb. Contacts"), omit = c("session_id", "role_id", "people_id"), single.row = TRUE, dep.var.labels = c("Log passed bills"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "tiny", omit.stat=c("LL","ser","f", "rsq"))

model1 <- lm(log(total_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1), data = authors_summary_with_committees)
model2 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1), data = authors_summary_with_committees)
model3 <- lm(log(LES) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1), data = authors_summary_with_committees)


stargazer(model1,model2,model3, add.lines=list(c('Chamber FE', 'Yes', 'Yes', 'Yes'), c('Session FE', 'Yes', 'Yes', 'Yes'), c('Legislator FE', 'No', 'No', 'No')), covariate.labels = 
            c("Minority party", "Chair committee", "Seniority", "Seniority sq.", "Log Lobb. Contacts"), omit = c("session_id", "role_id", "people_id"), single.row = TRUE, dep.var.labels = c("Log sponsored bills", "Log passed bills", "LES"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "small", omit.stat=c("LL","ser","f", "rsq"))


model1 <- lm(log(total_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1) + as.factor(people_id), data = authors_summary_with_committees)
model2 <- lm(log(passed_bills+0.01) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1)+ as.factor(people_id), data = authors_summary_with_committees)
model3 <- lm(log(LES) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + is_chair  + terms + I(terms^2) + log(parties_lobbied_id+1)+ as.factor(people_id), data = authors_summary_with_committees)


stargazer(model1,model2,model3, add.lines=list(c('Chamber FE', 'Yes', 'Yes', 'Yes'), c('Session FE', 'Yes', 'Yes', 'Yes'), c('Legislator FE', 'Yes', 'Yes', 'Yes')), covariate.labels = 
            c("Minority party", "Chair committee", "Seniority", "Seniority sq.", "Log Lobb. Contacts"), omit = c("session_id", "role_id", "people_id"), single.row = TRUE, dep.var.labels = c("Log sponsored bills", "Log passed bills", "LES"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "small", omit.stat=c("LL","ser","f", "rsq"))


model5 <- lm(log(total_bills+1) ~  log(legislative_amt+1) + log(parties_lobbied_id+1)  +as.factor(people_id), data = authors_summary_with_committees)
model6 <- lm(log(passed_bills+1) ~log(legislative_amt+1) + log(parties_lobbied_id+1)+as.factor(people_id), data = authors_summary_with_committees)

pdata_frame <- pdata.frame(authors_summary_with_committees, index = c("people_id"))

# Fit fixed effects model for log(total_bills+1)
fixed_effects_model_total_bills <- plm(log(total_bills+1) ~ log(legislative_amt+1) + log(parties_lobbied_id+1), 
                                       data = pdata_frame, model = "within")

# Fit fixed effects model for log(passed_bills+1)
fixed_effects_model_passed_bills <- plm(log(passed_bills+1) ~ log(legislative_amt+1) + log(parties_lobbied_id+1), 
                                        data = pdata_frame, model = "within")

# Summary of fixed effects model for log(total_bills+1)
summary(fixed_effects_model_total_bills)

# Summary of fixed effects model for log(passed_bills+1)
summary(fixed_effects_model_passed_bills)

histogram_legislative_amt <- ggplot(authors_summary_with_committees, aes(x = legislative_amt/1000)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(#title = "Histogram of log(legislative_amt+1)",
       x = "Expenditures in legislative staff (1000s $)",
       y = "Frequency")

ggsave(paste0(figures_subfolder, "histogram_legislative_amt.png"), width = 8, height = 6, units = "in", dpi = 300)
# Create histogram for log(parties_lobbied_id+1)
histogram_parties_lobbied_id <- ggplot(authors_summary_with_committees, aes(x = parties_lobbied_id.x +1)) +
  geom_histogram(binwidth = 180, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(#title = "Histogram of log(parties_lobbied_id+1)",
       x = "Total direct lobbying contacts",
       y = "Frequency")

# Save the histograms
ggsave(paste0(figures_subfolder, "histogram_lobbying_contacts.png"), width = 8, height = 6, units = "in", dpi = 300)


histogram_parties_lobbied_id <- ggplot(authors_summary_with_committees, aes(x = parties_lobbied_id.y+1)) +
  geom_histogram(binwidth = 50, fill = "lightgreen", color = "black", alpha = 0.7) +
  labs(#title = "Histogram of log(parties_lobbied_id+1)",
    x = "Total direct lobbying contacts (not associated with a bill id)",
    y = "Frequency")

# Save the histograms
ggsave(paste0(figures_subfolder, "histogram_lobbying_contacts_before.png"), width = 8, height = 6, units = "in", dpi = 300)





model1 <- lm(log(passed_bills+1) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + freshman + is_chair + terms.x, data = authors_summary_with_committees)
model2 <- lm(log(passed_bills+1) ~ as.factor(role_id) + as.factor(party_id) + as.factor(session_id) + freshman + is_chair + terms.x +  log(legislative_amt+1) + log(legislative_amt+1) + log(parties_lobbied_id+1), data = authors_summary_with_committees)


stargazer(model1,model2, add.lines=list(c('Session fixed effects', 'Yes', 'Yes')), covariate.labels = 
            c("Senator", "Minority party","Chair committee", "Seniority", "Log Legislative expenditures", "Log Lobbying Contacts", "(Intercept)"), omit = "session_id", single.row = TRUE, dep.var.labels = c("Bill sponsorship"),
          no.space = TRUE, column.sep.width = "3pt",
          font.size = "small", omit.stat=c("LL","ser","f", "rsq"))





model <- lm(LES ~ as.factor(party_id) + is_chair + terms_role, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 2,])
summary(model)

model <- lm(LES ~ as.factor(role_id) + as.factor(party_id)  + as.factor(session_id) + is_chair +  freshman + terms.x + log(parties_lobbied_id + 1)  + log(legislative_amt +1 ) , data =  authors_summary_with_committees)

model <- lm(LES ~ log(parties_lobbied_id + 1) + as.factor(role_id) + as.factor(party_id)  + as.factor(session_id) + freshman + terms, data =  authors_summary_with_committees)


model <- lm(LES ~ as.factor(party_id) + is_chair + terms_role, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 2,])
summary(model)


model <- lm(LPS ~ as.factor(party_id)  + is_chair + terms_role + intensity_contacts, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 1,])
summary(model)


model <- lm(LES ~ as.factor(year) + as.factor(party_id) + is_chair + terms_role +distinct_committee_count + I(legislative_experience/100) + I(log(intensity_contacts)/100), data = authors_summary_with_committees[authors_summary_with_committees$role_id == 1 & authors_summary_with_committees$year > 2018,])
summary(model)

authors_summary_with_committees[is.na(authors_summary_with_committees$intensity_expenditures) | authors_summary_with_committees$intensity_expenditures == 0, 'intensity_expenditures'] <- 0.1
authors_summary_with_committees[is.na(authors_summary_with_committees$legislative_amt) | authors_summary_with_committees$legislative_amt == 0, 'legislative_amt'] <- 0.1



panel_data <- pdata.frame(authors_summary_with_committees, index = c("session_id","people_id"))

# Perform fixed effects regression
model <- plm(weighted_total_bills ~ as.factor(role_id) + as.factor(party_id) + distinct_committee_count + chair_count + terms,
             data = panel_data, model = "within")

model <- glm(weighted_total_bills ~ as.factor(party_id) + as.factor(session_id) + as.factor(is_chair) + intensity_contacts,
             familiy = poisson, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 2,])

summary(model)
model <- glm(weighted_passed_bills ~ as.factor(party_id) + chair_count + terms_role + intensity_contacts*factor(party_id),
            family = poisson, data = authors_summary_with_committees[authors_summary_with_committees$role_id == 2,])

# View the summary of the Poisson regression model
summary(model)

# View the summary of the Poisson regression model
summary(model)

authors_expenses <- authors_summary %>%
  left_join(assembly_expenses, by = c("people_id", "session_id")) %>%
  filter(role_id == 1)

ggplot(authors_summary_temp, aes(x = legislative_amt)) +
  geom_point(aes(y = weighted_ratio_passed_authored, , color = factor(party_id))) +
  scale_color_manual(values = c("blue", 'red'), 
                     name = "Chamber/Party", 
                     labels = c("Rep (D)", "Rep (R)"))  +
  labs(title = "Effectiveness of lawmakers in each session (only lobbied bills)", x = "Number of bills introduced by lawmaker in session", y = "Ratio passed/introduced lobbied bills") +
  theme_minimal()


# # Create scatter plots for role_id == 1
# ggplot(authors_summary_temp, aes(x = weighted_lobbied_not_authored_bills)) +
#   geom_point(aes(y = weighted_ratio_passed_lobbied_not_authored, color = ifelse(role_id == 1, ifelse(party_id==1,"blue", "red"), ifelse(party_id == 1, "cyan", "pink")))) +
#   #geom_point(aes(y = authors_summary[authors_summary$role_id == 2,weighted_authored_bills], color = ifelse(party_id == 1, "cyan", "pink"))) +
#   scale_color_manual(values = c("blue", "red", "cyan", "pink"), name = "Chamber/Party", labels = c("Rep (D)", "Rep (R)", "Sen (D)", "Sen (R)")) +
#   labs(title = "Effectiveness of lawmakers in each session (only lobbied bills)", x = "Number of bills introduced by lawmaker in session", y = "Ratio passed/introduced lobbied bills") +
#   theme_minimal()
# 
# ggsave(paste0(figures_subfolder, "effectiveness_lobbying_all_lawmakers.png"), width = 8, height = 6, units = "in", dpi = 300)

sponsors_clients_lobbyist_contacts_table$retained_flag <- as.logical(sponsors_clients_lobbyist_contacts_table$retained_flag)

result_summary_ <- sponsors_clients_lobbyist_contacts_table %>%
  group_by(bill_id_new) %>%
  summarize(
    distinct_principal_lobbyist = n_distinct(principal_lobbyist),
  )
result_summary <- sponsors_clients_lobbyist_contacts_table %>%
  group_by(people_id) %>%
  summarize(
    distinct_principal_lobbyist = n_distinct(principal_lobbyist),
    distinct_beneficial_client = n_distinct(beneficial_client),
    distinct_principal_lobbyist_retained = n_distinct(principal_lobbyist[retained_flag == TRUE]),
    distinct_beneficial_client_retained = n_distinct(beneficial_client[retained_flag == TRUE]),
    distinct_principal_lobbyist_employed = n_distinct(principal_lobbyist[retained_flag == FALSE]),
    distinct_beneficial_client_employed = n_distinct(beneficial_client[retained_flag == FALSE])
  )

# Compute the average and standard deviation of distinct beneficial_client per principal_lobbyist
result_stats <- sponsors_clients_lobbyist_contacts_table %>%
  group_by(people_id, principal_lobbyist) %>%
  summarize(
    avg_beneficial_client = mean(n_distinct(beneficial_client)),
    sd_beneficial_client = sd(n_distinct(beneficial_client)),
    avg_beneficial_client_retained = mean(n_distinct(beneficial_client[retained_flag == TRUE])),
    sd_beneficial_client_retained = sd(n_distinct(beneficial_client[retained_flag == TRUE])),
    avg_beneficial_client_employed = mean(n_distinct(beneficial_client[retained_flag == FALSE])),
    sd_beneficial_client_employed = sd(n_distinct(beneficial_client[retained_flag == FALSE]))
  )

# Merge the two result sets on people_id
final_result <- merge(result_summary, result_stats, by = "people_id")

bills_principal_lobbyist <- sponsors_clients_lobbyist_contacts_table %>%
  group_by(people_id, principal_lobbyist) %>%
  summarize(
    num_bills = n(),
    sum_retained_flag = sum(retained_flag)
  )

# Create a table for (people_id, beneficial_client) with the number of bills and sum of retained_flag
bills_beneficial_client <- sponsors_clients_lobbyist_contacts_table %>%
  group_by(people_id, beneficial_client) %>%
  summarize(
    num_bills = n(),
    sum_retained_flag = sum(retained_flag)
  )


# Calculate shares for bills_principal_lobbyist
bills_principal_lobbyist <- bills_principal_lobbyist %>%
  group_by(people_id) %>%
  mutate(share = num_bills / sum(num_bills))

# Calculate shares for bills_beneficial_client
bills_beneficial_client <- bills_beneficial_client %>%
  group_by(people_id) %>%
  mutate(share = num_bills / sum(num_bills))

# Calculate the sum of squares (HHI)
hhi_principal_lobbyist <- bills_principal_lobbyist %>%
  group_by(people_id) %>%
  summarize(hhi = sum(share^2))

hhi_beneficial_client <- bills_beneficial_client %>%
  group_by(people_id) %>%
  summarize(hhi = sum(share^2))

# Merge the two dataframes on people_id
final_hhi <- merge(hhi_principal_lobbyist, hhi_beneficial_client, by = "people_id")

final_hhi <- merge(final_hhi, authors_summary, by = "people_id")



committees_table$is_matched_bill <- as.logical(committees_table$is_matched_bill)
committees_table$is_bill_lobbyist_lawmaker_matched <- as.logical(committees_table$is_bill_lobbyist_lawmaker_matched)
committees_table$passed <- as.logical(committees_table$passed)
committees_table$passes_first_referral <- as.logical(committees_table$passes_first_referral)

columns_to_replace_na <- c(
  "total_contacts",
  "total_member_contacts",                              
  "sum_sponsors_chamber_1_type_1",
  "sum_sponsors_chamber_1_type_2" ,                       
  "sum_sponsors_chamber_2_type_1",
  "sum_sponsors_chamber_2_type_2",                            
  "sum_committee_members_chamber_1_type_1",
  "sum_committee_members_chamber_1_type_2",                 
  "sum_committee_members_chamber_2_type_1",
  "sum_committee_members_chamber_2_type_2",
  "sum_committee_members_chamber_1_type_1_excluding_sponsors",
  "sum_committee_members_chamber_1_type_2_excluding_sponsors",
  "sum_committee_members_chamber_2_type_1_excluding_sponsors",
  "sum_committee_members_chamber_2_type_2_excluding_sponsors",
  "sum_committee_chamber_1_type_1",
  "sum_committee_chamber_1_type_2",                           
  "sum_committee_chamber_2_type_1",
  "sum_committee_chamber_2_type_2",                           
  "sum_member_chamber_1_excluding_sponsors_committee_members",
  "sum_member_chamber_2_excluding_sponsors_committee_members",
  "sum_member_chamber_1",
  "sum_member_chamber_2",                                     
  "sum_is_chamber_1",
  "sum_is_chamber_2"
)

committees_table <- committees_table %>%
  mutate(across(all_of(columns_to_replace_na), ~replace_na(., 0)))

committees_table$lobbied_committee <- (committees_table$sum_committee_members_chamber_1_type_1 > 0) | (committees_table$sum_committee_chamber_1_type_1 > 0)

committees_summary <- committees_table %>%
  group_by(committee_chamber_1_type_1, chamber_1, session_id, committee_size) %>%
  summarise(
    total_bills = n(),
    passed_bills = sum(passed),
    passed_first_bills = sum(passes_first_referral),
    lobbied_bills = sum(lobbied_committee),
    lobbied_passed_bills = sum(lobbied_committee & passed),
    lobbied_passed_first_bills = sum(lobbied_committee & passes_first_referral),
    lobbied_passed_passed_first_bills = sum(lobbied_committee & passed & passes_first_referral),
    passed_passed_first_bills = sum(passed & passes_first_referral),
    #authored_bills = sum(is_bill_lobbyist_lawmaker_matched),
    #authored_passed_bills = sum(is_bill_lobbyist_lawmaker_matched & passed),
    #ratio_passed_first = lobbied_bills/committee_size,
    ratio_passed_total = passed_bills/total_bills,
    ratio_lobbied_total = lobbied_bills/total_bills,
    ratio_passed_first_total = passed_first_bills/total_bills,
    ratio_passed_lobbied = lobbied_passed_bills/lobbied_bills,
    ratio_passed_first_lobbied = lobbied_passed_passed_first_bills/lobbied_bills,
    ratio_lobbied_passed = lobbied_passed_bills/passed_bills,
    ratio_lobbied_passed_first = lobbied_passed_first_bills/passed_first_bills,
    ratio_passed_passed_first = passed_passed_first_bills/passed_first_bills
    #ratio_authored_total = authored_bills/total_bills,
    #ratio_passed_authored = authored_passed_bills/authored_bills
    )
  
ggplot(committees_summary, aes(x = ratio_passed_first_lobbied)) +
  geom_point(aes(y = ratio_passed_first_total), color = "red") +
  #geom_point(aes(y = ratio_lobbied_total), color = "blue") +
  scale_color_identity() +
  labs(title = "Scatter plot for role_id == 1", x = "Total Bills", y = "weighted_ratio_authored_total", color = "Party ID") +
  theme_minimal()

ggplot(committees_summary, aes(x = ratio_passed_first_lobbied)) +
  #geom_point(aes(y = passed_first_bills/committee_size), color = "red") +
  geom_point(aes(y = ratio_passed_passed_first), color = "blue") +
  scale_color_identity() +
  labs(title = "Scatter plot for role_id == 1", x = "Total Bills", y = "weighted_ratio_authored_total", color = "Party ID") +
  theme_minimal()
