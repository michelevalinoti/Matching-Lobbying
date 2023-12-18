library(dplyr)
library(tidyverse)
library(modelsummary)

machine <- '/Users/michelev/Dropbox/lobbying'
ethics_subfolder <- file.path(machine, 'data/NYS_Ethics_Commission/')
legi_subfolder <- file.path(machine, 'data/NYS_LegiScan/')
df_subfolder <- file.path(machine, 'dataframes/')
tables_subfolder <- file.path(machine, 'tables/')
latex_tables_subfolder <- file.path(machine, 'latex_tables/')
client_df <- read.csv(paste(ethics_subfolder, 'client_list_all_sessions.csv', sep = ""), row.names = 1)
lobbyists_df <- read.csv(paste(ethics_subfolder, 'lobbyist_dataset_all_sessions.csv', sep = ""), row.names = 1)
lobbyists_df <- read.csv(paste(ethics_subfolder, 'lobbyist_dataset_all_sessions.csv', sep = ""), row.names = 1)
subjects_df <- read.csv(paste(ethics_subfolder, 'subjects_lobbied_all_sessions.csv', sep = ""), row.names = 1)
event_table  <- read.csv(paste(tables_subfolder, 'event_table.csv', sep = ""))


# Select rows where filing_semester is not NA
filtered_subjects_df <- subjects_df[subjects_df$filing_type == 'CLIENT SEMI-ANNUAL', ]

# Merge with lobbyists_df on principal_lobbyist_id
merged_df <- merge(filtered_subjects_df, lobbyists_df[, c("principal_lobbyist_id", "contractual_client_id", "year", "retained_flag")], by = c("principal_lobbyist_id", "contractual_client_id", "year"), all.x = TRUE)
merged_df$retained_flag = as.logical(merged_df$retained_flag)
merged_df <- merged_df %>%
  group_by(principal_lobbyist) %>%
  mutate(
    retained_flag = ifelse(
      n_distinct(beneficial_client) > 1,
      TRUE,
      ifelse(
        is.na(retained_flag) & n_distinct(beneficial_client) == 1,
        FALSE,
        retained_flag
      )
    )
  ) %>%
  ungroup()

result_client <- merged_df %>% 
  group_by(beneficial_client) %>% 
  summarize(
    avg_total_compensation = sum(total_compensation) / n_distinct(interaction(year, filing_semester))*2/1000,
    avg_total_compensation_retained = sum(total_compensation[retained_flag]) / n_distinct(interaction(year, filing_semester)[retained_flag])*2/1000,
    avg_total_compensation_employed = sum(total_compensation[!is.na(retained_flag) & retained_flag == FALSE]) / n_distinct(interaction(year, filing_semester)[!is.na(retained_flag) & retained_flag == FALSE])*2/1000,
    has_retained_flag_true = any(retained_flag == TRUE),
    has_retained_flag_false = any(retained_flag == FALSE),
    total_distinct_lobbyists = n_distinct(principal_lobbyist),
    #total_distinct_lobbyists_retained = n_distinct(principal_lobbyist[retained_flag == TRUE]),
    #total_distinct_lobbyists_employed = n_distinct(principal_lobbyist[retained_flag == FALSE]),
    concentration_of_main_issues = sum((table(main_issue) / sum(table(main_issue)))^2)
  ) %>% ungroup()

stargazer(
  as.data.frame(result_client),
  title = "Summary Statistics",
  label = "tab:summary",
  summary.stat = c("min", "median", "mean", "max"),
  covariate.labels = c(
    "Avg. expend./year (1000s, all)",
    "Avg. expend./year (1000s, retained)",
    "Avg. expend./year (1000s, employed)",
    "Has at least 1 retained lobbyist",
    "Has in-house lobbyist",
    "Distinct lobbyists (all)",
    #"Distinct lobbyists (retained)",
    #"Distinct lobbyists (employed)",
    "HHI of main issues"
  ),
  out = paste0(latex_tables_subfolder, 'clients_summary.tex')
)

result_lobbyist <- merged_df %>% 
  group_by(principal_lobbyist) %>%
  summarize(
    avg_total_compensation = sum(total_compensation) / n_distinct(interaction(year,filing_semester))*2,
    #avg_total_compensation_retained = sum(total_compensation[retained_flag]) / n_distinct(interaction(year, filing_semester)[retained_flag])*2,
    #avg_total_compensation_employed = sum(total_compensation[!is.na(retained_flag) & retained_flag == FALSE]) / n_distinct(interaction(year, filing_semester)[!is.na(retained_flag) & retained_flag == FALSE])*2,
    avg_client_compensation = sum(total_compensation) / n_distinct(interaction(year, filing_semester), contractual_client)*2,
    #avg_client_compensation_retained = sum(total_compensation[retained_flag]) / n_distinct(interaction(year, filing_semester), contractual_client)[retained_flag]*2,
    #avg_client_compensation_employed = sum(total_compensation[!is.na(retained_flag) & retained_flag == FALSE]) / n_distinct(interaction(year, filing_semester), contractual_client)[!is.na(retained_flag) & retained_flag == FALSE]*2,
    avg_retained_flag_true = ifelse(mean(retained_flag == TRUE, na.rm = TRUE) >= 0.5, 1, 0),
    avg_retained_flag_false = 1 - avg_retained_flag_true,
    total_distinct_clients = n_distinct(beneficial_client),
    concentration_of_main_issues = sum((table(main_issue) / sum(table(main_issue)))^2)
  ) %>% ungroup()

result_lobbyist <- result_lobbyist %>%
  mutate(`Avg. comp./year (1000s)` = avg_total_compensation / 1000,
         `Avg. comp./year/client (1000s)` = avg_client_compensation/1000,
         `Distinct clients` = total_distinct_clients,
         `HHI of issues` = concentration_of_main_issues,
         Type = ifelse(avg_retained_flag_true == TRUE, 'Retained', 'Employed')) %>%
  select(`Avg. comp./year (1000s)`,
         `Avg. comp./year/client (1000s)`,
         `Distinct clients`,
         Type,
         `HHI of issues`)

datasummary( Type*(`Avg. comp./year (1000s)` +
            `Avg. comp./year/client (1000s)` +
            `Distinct clients` +
            `HHI of issues`)~1*(N + Median + Mean  + SD), 
            data = result_lobbyist, output = paste0(latex_tables_subfolder, 'lobbyists_summary.tex'))

result_lobbyist <- as.data.frame(result_lobbyist)
stargazer(
  result_lobbyist ~ retained_flag,
  title = "Summary Statistics",
  label = "tab:summary",
  summary.stat = c("min", "median", "mean", "max"),
  covariate.labels = c(
    "Avg. yearly compensation (in 1000s, all)",
    "Avg. yearly compensation (in 1000s, retained)",
    "Avg. yearly compensation (in 1000s, employed)",
    "Avg. yearly compensation per client(in 1000s, all)",
    "Avg. yearly compensation (in 1000s, retained)",
    "Avg. yearly compensation (in 1000s, employed)",
    "Has at least 1 retained lobbyist",
    "Has in-house lobbyist",
    "Total distinct lobbyists",
    "HHI of main issues"
  )
)




parties_tot <- read.csv(paste(ethics_subfolder, 'parties_lobbied_merged.csv', sep = ""), row.names = 1)
parties_tot <- parties_tot[(parties_tot$communication == 'DIRECT LOBBYING') | (parties_tot$communication == 'BOTH DIRECT AND GRASSROOTS'),]
parties_tot <- parties_tot[(parties_tot$filing_type == 'BI-MONTHLY') | (parties_tot$filing_type == 'BI-MONTHLY AMENDMENT'),]

lobbyist_dataset <- read.csv(paste(ethics_subfolder, 'lobbyist_dataset_all_sessions.csv', sep = ""), row.names = 1)
subjects_lobbied <- read.csv(paste(ethics_subfolder, 'subjects_lobbied_all_sessions.csv', sep = ""), row.names = 1)

# for summary statistics of the complete dataset, focus on unique bimonthly reports where (period, lobbyist, client, govt body, focus level, focus) vary
#report_columns <- c('year', 'party_name', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client', 'govt_body', 'activity', 'focus')
report_columns <- c('year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client', 'activity')
activities_df <- parties_tot[!duplicated(parties_tot[, report_columns]), c(report_columns, c('main_issue', 'retained_flag', 'total_compensation', 'annual_compensation'))]
activities_df <- activities_df[activities_df$retained_flag!="",]

#parties_contacts <- parties_contacts %>%
#  left_join(lobbyist_dataset %>% select(principal_lobbyist_id, beneficial_client_id, retained_flag), by = c('principal_lobbyist_id', 'beneficial_client_id'))
activities_df$activities_df <- as.logical(activities_df$retained_flag)

activities_df <- activities_df[activities_df$retained_flag==TRUE,]

sum_compensation <- activities_df %>% 
  group_by(main_issue, activity) %>%
  summarize(sum_compensation = sum(total_compensation, na.rm =TRUE), total_reports = n_distinct(year,filing_bimonth,principal_lobbyist, na.rm=TRUE))
  
pivot_table <- activities_df %>%
  group_by(main_issue, activity, principal_lobbyist) %>%
  summarize(reports = n_distinct(year,filing_bimonth, na.rm=TRUE), compensation = sum(total_compensation)) %>%
  left_join(sum_compensation[,c('main_issue', 'activity','sum_compensation', 'total_reports')], by = c('main_issue', 'activity')) %>%
  mutate(ratio_compensation = (compensation/sum_compensation)**2, ratio_reports = (reports/total_reports)**2) %>%
  group_by(main_issue,activity) %>%
  summarize(concentration = sum(ratio_compensation), reports_concentration = sum(ratio_reports))

  mutate(ratio_compensation = (total_compensation/sum_compensation)**2)
  group_by(main_issue, activity) %>%
  summarize(concentration = sum(sum_compensation,))
temp_df <- 
  total_df <- df %>%
  mutate(A = 'TOTAL')

# Append total_df to df
result_df <- bind_rows(df, total_df)
  
  
avg_lobbyist <- parties_contacts %>%
  group_by(activity, govt_body, principal_lobbyist) %>%
  summarize(no_distinct_issues_lobbyist = n_distinct(focus)) %>%
  group_by(activity, govt_body) %>%
  summarize(avg_issues_lobbyist = n_distinct(no_distinct_issues_lobbyist))
avg_client <- parties_contacts %>%
  group_by(activity, govt_body, beneficial_client) %>%
  summarize(no_distinct_issues_client = n_distinct(focus)) %>%
  group_by(activity, govt_body) %>%
  summarize(avg_issues_client = n_distinct(no_distinct_issues_client))

pivot_table <- parties_contacts %>%
  group_by(activity, govt_body) %>%
  summarize(no_distinct_lobbyist_client = n_distinct(paste(principal_lobbyist, beneficial_client, sep = "-")),
            no_distinct_lobbyists = n_distinct(principal_lobbyist),
            no_distinct_clients = n_distinct(beneficial_client),
            avg_retained = mean(retained_flag, na.rm = TRUE)) %>%
  left_join(avg_lobbyist, by = c('activity', 'govt_body')) %>%
  left_join(avg_client, by = c('activity', 'govt_body'))
