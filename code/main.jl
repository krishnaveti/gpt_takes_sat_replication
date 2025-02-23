using Pkg
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("StatsPlots")
Pkg.add("CSV")
Pkg.add("XLSX")
Pkg.add("Distributions")
Pkg.add("StatsBase")

# Minor variations in Random Forest predictions for figure 5 are expected due to random seed differences at the time of the run.

using XLSX, DataFrames, CSV, StatsBase
using Distributions, Plots, StatsPlots

dir = "../data"

# Figure 1
pre_conv = CSV.read(dir * "/SAT_Math_Score_Conversion.csv", DataFrame)
post_conv = CSV.read(dir * "/SAT_Math_Score_Conversion_New.csv", DataFrame)
old_new_conv = CSV.read(dir * "/sat_conversion_table_complete.csv", DataFrame)
predict_new_sat(old_sat; int=43.956, slope=0.970) = round(Int, int + slope * old_sat)
clean_string(s) = replace(lowercase(s), r"[^a-z]" => "")

post_stud_mean = reverse([508, 521, 528, 523, 528, 531, 527]) # Post Student
post_stud_sd = reverse([122, 120, 120, 117, 117, 114, 107])
n_post = reverse([1913742, 1737678, 1509133, 2198460, 2220087, 2136539, 1715481])
pre_stud_mean = [515, 515, 516, 514, 514, 514, 513, 511, 508] # Pre Students
pre_stud_sd = [116, 116, 116, 117, 117, 118, 120, 120, 121]
n_pre = [1518859, 1530128, 1547990, 1647123, 1664479, 1660047, 1672395, 1698521, 1637589]

pre_student = predict_new_sat.(pre_stud_mean)

# Data for Post and Pre Students
y_post = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
# post_stud_mean = [527, 531, 528, 523, 528, 521, 508]
ci_post = post_stud_sd ./ sqrt.(n_post) .* 1.96
upper_post = post_stud_mean .+ ci_post
lower_post = post_stud_mean .- ci_post

y_pre = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
# pre_stud_mean = [515, 515, 516, 514, 514, 514]
ci_pre = pre_stud_sd ./ sqrt.(n_pre) .* 1.96
upper_pre = pre_stud_mean .+ ci_pre
lower_pre = pre_stud_mean .- ci_pre

# Create a new plot
p = plot(legend=:topright, ylims=[500, 560], yticks=(490:10:560), dpi=600, xticks=(2008:2:2024))

# Plot for Post Students with confidence intervals
plot!(p, y_post, upper_post, ribbon=(ci_post, ci_post), m=:o, c=:red, ms=3, lw=0.5, fillalpha=0.6, label="Mean Score")
# scatter!(p, y_post, post_stud_mean, label="Post Students Mean", color=:blue, marker=:circle)

# Plot for Pre Students with confidence intervals
plot!(p, y_pre, upper_pre, ribbon=(ci_pre, ci_pre), m=:x, c=:green, ls=:dot, ms=3, fillalpha=0.6, lw=0.5, label="Mean Score")
# scatter!(p, y_pre, pre_stud_mean, label="Pre Students Mean", color=:red, marker=:circle)
upper_pre_old = round.(Int, upper_pre ./ 10) .* 10
upper_pre_conv = old_new_conv[[findall(upper_pre_old[i] .== old_new_conv[:, 1])[1] for i in 1:9], 2]

plot!(p, y_pre, predict_new_sat.(upper_pre), ribbon=(ci_pre, ci_pre), lw=0.5, m=:o, c=:red, ms=3, fillalpha=0.6, label="Mean SAT Converted Score")

plot!(p, y_pre, upper_pre_conv, ribbon=(ci_pre, ci_pre), m=:square, ms=3, ls=:dash, lw=0.5, c=:blue, fillalpha=0.6, label="Mean SAT Concordance Score")
vline!([2016, 2017], ylims=[500, 560], c=:black, lw=2, fillrange=[2016, 2017], alpha=0.6, label=false)
annotate!((2016 + 2017) / 2, (490 + 560) / 2, text("SAT Format Change", :black, :center, 11, :bold, rotation=90))

# Adding labels and title
xlabel!(p, "Year")
ylabel!(p, "Mean Score")
title!(p, "Mean SAT Scores by Year")

savefig(p, "../results/fig1.png")

# Figure 2
gpt_df = CSV.read(dir * "/gpt4_january_scores_2008_2023.csv", DataFrame)

gpt_data = combine(groupby(gpt_df, :year, sort=true), [:scaled_score => (x -> round(mean(x))) => :gpt_score,
    :scaled_score => (x -> round(std(x)) / sqrt(50)) => :std_error,
    :mcq_prop => (x -> round(mean(x), digits=3)) => :mcq_prop,
    :ans_prop => (x -> round(mean(x), digits=3)) => :ans_prop,
    :mcq_prop => (x -> round(std(x) / sqrt(50), digits=3)) => :mcq_se,
    :ans_prop => (x -> round(std(x) / sqrt(50), digits=3)) => :ans_se,
    :raw_score => (x -> round(mean(x), digits=3)) => :raw_score,
    :raw_score => (x -> round(std(x) / sqrt(50), digits=3)) => :raw_score_se,
    :prop => (x -> round(mean(x), digits=3)) => :raw_prop,
    :prop => (x -> round(std(x) / sqrt(50), digits=3)) => :raw_se]
)

p = plot(dpi=600, xlabel="Year", ylabel="Mean SAT Score")
plot!(p, 2008:2023, gpt_data[:, 2], label="LLM Agent", yerror=gpt_data[:, 3], lw=1, m=:o, ms=1, ls=:dot, color=:blue)

savefig("../results/fig2a.png")

p = plot(dpi=600, xlabel="Year", ylabel="Correct Answers")
plot!(p, 2008:2023, gpt_data[:, 8], label="LLM Agent", yerror=gpt_data[:, 9], lw=1, m=:o, ms=1, ls=:dot, color=:blue)

savefig("../results/fig2b.png")

p = plot(dpi=600, xlabel="Year", ylabel="Proportion/Percentage of Correct Answers")
plot!(p, 2008:2023, gpt_data[:, 10], label="LLM Agent", yerror=gpt_data[:, 11], lw=1, m=:o, ms=1, ls=:dot, color=:blue)

savefig("../results/fig2c.png")

# Figure 3
national_data = DataFrame(
    year=[2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    student_score=[predict_new_sat.([515, 515, 516, 514, 514, 514, 513, 511, 508]); reverse([508, 521, 528, 523, 528, 531, 527])],
    standard_deviation=[[116, 116, 116, 117, 117, 118, 120, 120, 121]; reverse([122, 120, 120, 117, 117, 114, 107])],
    n=[[1518859, 1530128, 1547990, 1647123, 1664479, 1660047, 1672395, 1698521, 1637589]; reverse([1913742, 1737678, 1509133, 2198460, 2220087, 2136539, 1715481])]
)
national_data.std_error = national_data.standard_deviation ./ sqrt.(national_data.n)

state_df = DataFrame(XLSX.readtable(dir * "/state_data_weighted_2.xlsx", "Sheet1"))

mass_df2 = DataFrame(XLSX.readtable(dir * "/mass_data_weighted.xlsx", "Sheet1"))
sort!(mass_df2, :year)

mass_schs = intersect([lowercase.(mass_df2.district[mass_df2.year.==i]) for i in unique(mass_df2.year)]...)

mass_df = mass_df2[[school in mass_schs for school in lowercase.(mass_df2[!, :district])], :]
mass_df.sat_math = ifelse.(mass_df.year .<= 2016, predict_new_sat.(mass_df.sat), mass_df.sat)

# State by State
state_data = combine(groupby(state_df, :year, sort=true), [:sat_math => (x -> round(mean(x))) => :student_score,
    :sat_math => (x -> round(std(x)) / sqrt(51)) => :std_error]
)
state_data.n = fill(51, size(state_data, 1))

# Massachusetts
mass_data = combine(groupby(mass_df, :year, sort=true), [:sat_math => (x -> round(mean(x))) => :student_score,
    :sat_math => (x -> round(std(x)) / sqrt(228)) => :std_error]
)
mass_data.n = fill(228, size(mass_data, 1))

# Difference in Means
diff = zeros(100000, 16)
diff_nat = zeros(100000, 16)
diff_nat2 = zeros(100000, 16)
diff_nat3 = zeros(100000, 16)
diff_g = zeros(100000, 16)
diff_state = zeros(100000, 16)
diff_mass = zeros(100000, 16)
for (i, y) in enumerate(2009:2023)
    gpt_base = gpt_data[gpt_data.year.==2008, :gpt_score][1]
    gpt_base_se = gpt_data[gpt_data.year.==2008, :std_error][1]

    national_base = national_data[national_data.year.==2008, :student_score][1]
    national_base_se = national_data[national_data.year.==2008, :std_error][1]
    national_base_n = national_data[national_data.year.==2008, :n][1]

    state_base = state_data[state_data.year.==2008, :student_score][1]
    state_base_se = state_data[state_data.year.==2008, :std_error][1]
    state_base_n = state_data[state_data.year.==2008, :n][1]

    mass_base = mass_data[mass_data.year.==2008, :student_score][1]
    mass_base_se = mass_data[mass_data.year.==2008, :std_error][1]
    mass_base_n = mass_data[mass_data.year.==2008, :n][1]

    gpt_mean = gpt_data[gpt_data.year.==y, :gpt_score][1]
    national_mean = national_data[national_data.year.==y, :student_score][1]
    national_mean_n = national_data[national_data.year.==y, :n][1]

    state_mean = state_data[state_data.year.==y, :student_score][1]
    state_mean_n = state_data[state_data.year.==y, :n][1]

    mass_mean = mass_data[mass_data.year.==y, :student_score][1]
    mass_mean_n = mass_data[mass_data.year.==y, :n][1]

    gpt_mean_se = gpt_data[gpt_data.year.==y, :std_error][1]
    national_mean_se = national_data[national_data.year.==y, :std_error][1]
    state_mean_se = state_data[state_data.year.==y, :std_error][1]
    mass_mean_se = mass_data[mass_data.year.==y, :std_error][1]

    diff_gpt = gpt_base .+ gpt_base_se .* rand(TDist(50 - 1), 100000) .- (gpt_mean .+ gpt_mean_se .* rand(TDist(50 - 1), 100000))

    diff_national = national_base .+ national_base_se .* rand(TDist(national_base_n - 1), 100000) .-
                    (national_mean .+ national_mean_se .* rand(TDist(national_mean_n - 1), 100000))

    diff_2 = state_base .+ state_base_se .* rand(TDist(state_base_n - 1), 100000) .-
             (state_mean .+ state_mean_se .* rand(TDist(state_mean_n - 1), 100000))

    diff_3 = mass_base .+ mass_base_se .* rand(TDist(mass_base_n - 1), 100000) .-
             (mass_mean .+ mass_mean_se .* rand(TDist(mass_mean_n - 1), 100000))

    diff_nat[:, i+1] = diff_national
    diff_nat2[:, i+1] = diff_2
    diff_nat3[:, i+1] = diff_3
    diff_g[:, i+1] = diff_gpt
    diff[:, i+1] = diff_gpt .- diff_national
    diff_state[:, i+1] = diff_gpt .- diff_2
    diff_mass[:, i+1] = diff_gpt .- diff_3
end

diff_mean = round.(digits=3, [mean(diff[:, i]) for i in 1:16])
diff_se = round.(digits=3, [std(diff[:, i]) for i in 1:16])

diff_nat_mean = round.(digits=3, [mean(diff_nat[:, i]) for i in 1:16])
nat_se = round.(digits=3, [std(diff_nat[:, i]) for i in 1:16])
nat_ci = nat_se .* 1.96

diff_state_mean = round.(digits=3, [mean(diff_nat2[:, i]) for i in 1:16])
nat2_se = round.(digits=3, [std(diff_nat2[:, i]) for i in 1:16])
nat2_ci = nat2_se .* 1.96

diff_mass_mean = round.(digits=3, [mean(diff_nat3[:, i]) for i in 1:16])
nat3_se = round.(digits=3, [std(diff_nat3[:, i]) for i in 1:16])
nat3_ci = nat3_se .* 1.96

diff_g_mean = round.(digits=3, [mean(diff_g[:, i]) for i in 1:16])
g_se = round.(digits=3, [std(diff_g[:, i]) for i in 1:16])
g_ci = g_se .* 1.96

nat_mean = diff_mean
ads_se = round.(digits=3, [std(diff[:, i]) for i in 1:16])
ads_ci = ads_se .* 1.96

state_mean = round.(digits=3, [mean(diff_state[:, i]) for i in 1:16])
ads_state_se = round.(digits=3, [std(diff_state[:, i]) for i in 1:16])
ads_state_ci = ads_state_se .* 1.96

mass_mean = round.(digits=3, [mean(diff_mass[:, i]) for i in 1:16])
ads_mass_se = round.(digits=3, [std(diff_mass[:, i]) for i in 1:16])
ads_mass_ci = ads_mass_se .* 1.96

p = plot(legend=:bottomleft, dpi=600, xticks=[(2008:3:2023); 2023], xlabel="Year", ylabel="Change from Baseline SAT Score")

# ADS using National Population Averages
plot!(p, 2008:2023, nat_mean, lw=0.5, m=:o, c=:red, yerror=(ads_ci, ads_ci), ms=1, fillalpha=0.4, title="National Population SAT Statistics", label="ADS")

plot!(p, 2008:2023, -diff_g_mean, lw=0.5, m=:x, c=:blue, yerror=(g_ci, g_ci), ms=1, fillalpha=0.4, label="GPT Performance")

plot!(p, 2008:2023, -diff_nat_mean, lw=0.5, m=:square, c=:green, yerror=(nat_ci, nat_ci), ms=1, fillalpha=0.4, label="Student Performance")
savefig(p , "../results/fig3d.png")

# ADS using State SAT Averages
p = plot(legend=:bottomleft, dpi=600, xticks=[(2008:3:2023); 2023], xlabel="Year", ylabel="Change from Baseline SAT Score")
plot!(p, 2008:2023, state_mean, lw=0.5, m=:o, c=:red, yerror=(ads_state_ci, ads_state_ci), ms=1, fillalpha=0.4, title="State Average National Sample", label="ADS")

plot!(p, 2008:2023, -diff_g_mean, lw=0.5, m=:x, c=:blue, yerror=(g_ci, g_ci), ms=1, fillalpha=0.4, label="GPT Performance")

plot!(p, 2008:2023, -diff_state_mean, lw=0.5, m=:square, c=:green, yerror=(nat2_ci, nat2_ci), ms=1, fillalpha=0.4, label="Student Performance")

savefig(p , "../results/fig3a.png")

# ADS using Mass SAT Averages
p = plot(legend=:bottomleft, dpi=600, xticks=[(2008:3:2023); 2023], xlabel="Year", ylabel="Change from Baseline SAT Score")
plot!(p, 2008:2023, mass_mean, lw=0.5, m=:o, c=:red, yerror=(ads_mass_ci, ads_mass_ci), ms=1, fillalpha=0.4, title="Massachusetts School District SAT Sample", label="ADS")

plot!(p, 2008:2023, -diff_g_mean, lw=0.5, m=:x, c=:blue, yerror=(g_ci, g_ci), ms=1, fillalpha=0.4, label="GPT Performance")

plot!(p, 2008:2023, -diff_mass_mean, lw=0.5, m=:square, c=:green, yerror=(nat3_ci, nat3_ci), ms=1, fillalpha=0.4, label="Student Performance")

savefig(p , "../results/fig3e.png")

# National
p = plot(legend=:bottomleft, dpi=600, xticks=[(2008:3:2023); 2023], xlabel="Year", ylabel="Change from Baseline SAT Score")
plot!(p, 2008:2023, nat_mean, lw=0.7, m=:o, c=:red, ms=3, fillalpha=0.7, title="Difference in Mean Scores Relative to Baseline", label="National Population")

# State by State
plot!(p, 2008:2023, state_mean, lw=0.7, ls=:dot, m=:x, c=:green, ms=3, fillalpha=0.7, label="State SAT Sample")

# Massachusetts
plot!(p, 2008:2023, mass_mean, lw=0.7, ls=:dash, m=:utriangle, c=:blue, ms=3, fillalpha=0.7, label="Massachusetts Districts")

savefig(p , "../results/fig3b.png")

# Race Data
# Load race data
data = XLSX.readtable(dir * "/race_gender_sat.xlsx", "race") |> DataFrame
data.std_error = data.sd ./ sqrt.(data.n)

data_2008_2016 = data[data.reported_year.<=2016, :]
data_2008_2016.sat = predict_new_sat.(data_2008_2016.sat)
data = vcat(data_2008_2016, data[data.reported_year.>2016, :])

# Group data by year and race
asian_data = combine(groupby(data[data.race.=="asian", :], :reported_year, sort=true),
    [:sat => mean => :student_score,
        :std_error => mean => :std_error,
        :n => mean => :n])

black_data = combine(groupby(data[data.race.=="black", :], :reported_year, sort=true),
    [:sat => mean => :student_score,
        :std_error => mean => :std_error,
        :n => mean => :n])

white_data = combine(groupby(data[data.race.=="white", :], :reported_year, sort=true),
    [:sat => mean => :student_score,
        :std_error => mean => :std_error,
        :n => mean => :n])

# Difference in Means Analysis
diff_asian = zeros(100000, 16)
diff_black = zeros(100000, 16)
diff_white = zeros(100000, 16)
diff_gpt = zeros(100000, 16)
diff_asian_gpt = zeros(100000, 16)
diff_black_gpt = zeros(100000, 16)
diff_white_gpt = zeros(100000, 16)
for (i, y) in enumerate(2009:2023)
    gpt_base = gpt_data[gpt_data.year.==2008, :gpt_score][1]
    gpt_base_se = gpt_data[gpt_data.year.==2008, :std_error][1]
    gpt_mean = gpt_data[gpt_data.year.==y, :gpt_score][1]
    gpt_mean_se = gpt_data[gpt_data.year.==y, :std_error][1]

    diff_gpt[:, i+1] = gpt_base .+ gpt_base_se .* rand(TDist(50 - 1), 100000) .- (gpt_mean .+ gpt_mean_se .* rand(TDist(50 - 1), 100000))

    # Asian data
    asian_base = asian_data[asian_data.reported_year.==2008, :student_score][1]
    asian_base_se = asian_data[asian_data.reported_year.==2008, :std_error][1]
    asian_n = asian_data[asian_data.reported_year.==2008, :n][1]
    asian_mean = asian_data[asian_data.reported_year.==y, :student_score][1]
    asian_mean_se = asian_data[asian_data.reported_year.==y, :std_error][1]
    asian_n_y = asian_data[asian_data.reported_year.==y, :n][1]

    diff_asian[:, i+1] = asian_base .+ asian_base_se .* rand(TDist(asian_n - 1), 100000) .-
                         (asian_mean .+ asian_mean_se .* rand(TDist(asian_n_y - 1), 100000))

    diff_asian_gpt[:, i+1] = diff_gpt[:, i+1] .- diff_asian[:, i+1]

    # Black data
    black_base = black_data[black_data.reported_year.==2008, :student_score][1]
    black_base_se = black_data[black_data.reported_year.==2008, :std_error][1]
    black_n = black_data[black_data.reported_year.==2008, :n][1]
    black_mean = black_data[black_data.reported_year.==y, :student_score][1]
    black_mean_se = black_data[black_data.reported_year.==y, :std_error][1]
    black_n_y = black_data[black_data.reported_year.==y, :n][1]

    diff_black[:, i+1] = black_base .+ black_base_se .* rand(TDist(black_n - 1), 100000) .-
                         (black_mean .+ black_mean_se .* rand(TDist(black_n_y - 1), 100000))

    diff_black_gpt[:, i+1] = diff_gpt[:, i+1] .- diff_black[:, i+1]

    # White data
    white_base = white_data[white_data.reported_year.==2008, :student_score][1]
    white_base_se = white_data[white_data.reported_year.==2008, :std_error][1]
    white_n = white_data[white_data.reported_year.==2008, :n][1]
    white_mean = white_data[white_data.reported_year.==y, :student_score][1]
    white_mean_se = white_data[white_data.reported_year.==y, :std_error][1]
    white_n_y = white_data[white_data.reported_year.==y, :n][1]

    diff_white[:, i+1] = white_base .+ white_base_se .* rand(TDist(white_n - 1), 100000) .-
                         (white_mean .+ white_mean_se .* rand(TDist(white_n_y - 1), 100000))

    diff_white_gpt[:, i+1] = diff_gpt[:, i+1] .- diff_white[:, i+1]
end

# Compute mean and confidence intervals for differences
diff_asian_gpt_mean = round.(digits=3, [mean(diff_asian_gpt[:, i]) for i in 1:16])
diff_asian_gpt_se = round.(digits=3, [std(diff_asian_gpt[:, i]) for i in 1:16])
diff_asian_gpt_ci = diff_asian_gpt_se .* 1.96

diff_black_gpt_mean = round.(digits=3, [mean(diff_black_gpt[:, i]) for i in 1:16])
diff_black_gpt_se = round.(digits=3, [std(diff_black_gpt[:, i]) for i in 1:16])
diff_black_gpt_ci = diff_black_gpt_se .* 1.96

diff_white_gpt_mean = round.(digits=3, [mean(diff_white_gpt[:, i]) for i in 1:16])
diff_white_gpt_se = round.(digits=3, [std(diff_white_gpt[:, i]) for i in 1:16])
diff_white_gpt_ci = diff_white_gpt_se .* 1.96

# Plotting
p = plot(legend=:bottomleft, dpi=600, xticks=[(2008:3:2023); 2023], xlabel="Year", title="GPT vs Racial Group Difference of Differences", ylabel="Change from Baseline SAT Score")

plot!(p, 2008:2023, diff_asian_gpt_mean, lw=0.5, m=:o, c=:red, ribbon=(diff_asian_gpt_ci, diff_asian_gpt_ci), ms=1, fillalpha=0.4, label="Asian vs GPT")
plot!(p, 2008:2023, diff_black_gpt_mean, lw=0.5, m=:x, c=:blue, ribbon=(diff_black_gpt_ci, diff_black_gpt_ci), ms=1, fillalpha=0.4, label="Black vs GPT")
plot!(p, 2008:2023, diff_white_gpt_mean, lw=0.5, m=:square, c=:green, ribbon=(diff_white_gpt_ci, diff_white_gpt_ci), ms=1, fillalpha=0.4, label="White vs GPT")

savefig(p , "../results/fig3c.png")

# Figure 5
all_qs = XLSX.readtable(dir * "/all_questions_embeddings_predictions_progress_bar_qnum.xlsx", "Sheet1") |> DataFrame
data_df_jan = DataFrame(XLSX.readtable(dir * "/gpt_answers_all_years_updated_2011_2020.xlsx", "Sheet1"))
data_ans = DataFrame(XLSX.readtable(dir * "/gpt_answers_anstype_all_years_scored_final_corrected_updated.xlsx", "Sheet1"))

mcqs_post = 45
mcqs_pre = 42

answ_pre = 54 - mcqs_pre
answ_post = 58 - mcqs_post

mcqs_pre_id = findall(all_qs.question_type[all_qs.year.<=2016] .== "mcq")
answ_pre_id = findall(all_qs.question_type[all_qs.year.<=2016] .== "answer")

mcqs_post_id = findall(all_qs.question_type[all_qs.year.>2016] .== "mcq")
answ_post_id = findall(all_qs.question_type[all_qs.year.>2016] .== "answer")

all_qs.code = parse.(
    Int64, string.(all_qs.year .+ 1) .*
           string.(all_qs.section_num) .*
           string.(all_qs.question_num)
)

data_df = data_df_jan

data_mcq = data_df[data_df.question_type.=="mcq", :]

data_mcq.clean_answer = clean_string.(data_mcq.gpt_answer)

data_mcq.mcq_scored = data_mcq.true_answer .== clean_string.(data_mcq.gpt_answer)

data_mcq.code = parse.(
    Int64, string.(data_mcq.reported_year) .*
           string.(data_mcq.section_num) .*
           string.(data_mcq.question_num)
)

data_mcq.label_prediction = Vector{String}(undef, size(data_mcq, 1))

for i in 1:size(all_qs, 1)
    data_mcq.label_prediction[findall(data_mcq.code .== all_qs.code[i])] .= all_qs.label_predictions[i]
end

mcq_correct = Int64[]
for y in 2007:2023
    if y < 2016
        comp_df = data_mcq[data_mcq.exam_year.==y, [:mcq_scored, :bootstrap_sample]]
        mcqs_sum = [count(comp_df.mcq_scored[1+(i-1)*mcqs_pre:i*mcqs_pre]) for i in unique(comp_df.bootstrap_sample)]
        append!(mcq_correct, mcqs_sum)
    else
        comp_df = data_mcq[data_mcq.exam_year.==y, [:mcq_scored, :bootstrap_sample]]
        mcqs_sum = [count(comp_df.mcq_scored[1+(i-1)*mcqs_post:i*mcqs_post]) for i in unique(comp_df.bootstrap_sample)]
        append!(mcq_correct, mcqs_sum)
    end
end

data_ans.code = parse.(
    Int64, string.(data_ans.reported_year) .*
           string.(data_ans.section_num) .*
           string.(data_ans.question_num)
)

data_ans.label_prediction = Vector{String}(undef, size(data_ans, 1))

for i in 1:size(all_qs, 1)
    data_ans.label_prediction[findall(data_ans.code .== all_qs.code[i])] .= all_qs.label_predictions[i]
end

answ_correct = Int64[]
for y in 2007:2023
    if y < 2016
        comp_df = data_ans[data_ans.exam_year.==y, [:ans_scored, :bootstrap_sample]]
        answ_sum = [sum(comp_df.ans_scored[1+(i-1)*answ_pre:i*answ_pre]) for i in unique(comp_df.bootstrap_sample)]
        append!(answ_correct, answ_sum)
    else
        comp_df = data_ans[data_ans.exam_year.==y, [:ans_scored, :bootstrap_sample]]
        answ_sum = [sum(comp_df.ans_scored[1+(i-1)*answ_post:i*answ_post]) for i in unique(comp_df.bootstrap_sample)]
        append!(answ_correct, answ_sum)
    end
end

total_correct = mcq_correct .+ answ_correct

df_mcq = data_mcq[:, [:exam_year, :mcq_scored, :label_prediction, :bootstrap_sample]]
df_mcq.scored = convert.(Int64, df_mcq.mcq_scored)
df_ans = data_ans[:, [:exam_year, :ans_scored, :label_prediction, :bootstrap_sample]]
df_ans.scored = convert.(Int64, df_ans.ans_scored)

df_all = vcat(df_mcq, df_ans, cols=:intersect)

gdf_all = groupby(df_all, [:exam_year, :bootstrap_sample, :label_prediction], sort=true)

# Question difficulty by year
all_scored = combine(gdf_all, :scored => mean => :proportion_scored)

all_scored_by_year = combine(groupby(all_scored, [:exam_year, :label_prediction], sort=true), [:proportion_scored => mean => :proportion_bootstrap_scored,
    :proportion_scored => std => :proportion_bootstrap_std])

all_scored_by_year.year = all_scored_by_year.exam_year .+ 1

# plots
plot_easy = all_scored_by_year[all_scored_by_year.label_prediction.=="easy", :][1:end-1, :]
plot_medium = all_scored_by_year[all_scored_by_year.label_prediction.=="medium", :][1:end-1, :]
plot_hard = all_scored_by_year[all_scored_by_year.label_prediction.=="hard", :][1:end-1, :]

p=plot(dpi=600, title="Proportion of Correct Answers by Question Type", xlabel="Year", ylabel="Correct Answered", legend=:bottomright)
plot!(plot_easy.year, plot_easy.proportion_bootstrap_scored, label="Easy", lw=2, m=:o, ms=3, ls=:dot, color=:blue)
plot!(plot_medium.year, plot_medium.proportion_bootstrap_scored, label="Medium", lw=2, m=:x, ls=:dash, ms=3, color=:green)
plot!(plot_hard.year, plot_hard.proportion_bootstrap_scored, label="Hard", lw=2, m=:square, ms=3, color=:red)
vline!([2014], label=false, color=:black, lw=2, ls=:dash)
annotate!((2011 + 2012) / 2, 0.7, text(" ← True Classes", :black, :center, 10))
annotate!((2017 + 2017) / 2, 0.3, text("Predicted Classes →", :black, :center, 10))
savefig(p , "../results/fig5a.png")

# Question Consistency by year
new_gdf = groupby(df_all, [:exam_year, :bootstrap_sample], sort=true)
new_gdf_exams = combine(new_gdf, :label_prediction => countmap => :label_counts)
new_gdf_exams.qcount = [fill(54, 9 * 50); fill(58, 8 * 50)]

[values(new_gdf_exams[i, :label_counts]) ./ new_gdf_exams[i, :qcount] for i in 1:850]

avg_diff = hcat([mean([values(new_gdf_exams[i, :label_counts]) ./ new_gdf_exams[i, :qcount] for i in 1:850][(i-1)*50+1:(i)*50]) for i in 1:17]...)
std_diff = hcat([std([values(new_gdf_exams[i, :label_counts]) ./ new_gdf_exams[i, :qcount] for i in 1:850][(i-1)*50+1:(i)*50]) for i in 1:17]...)
# Plot
p=plot(dpi=600, title="Proportion of Question Type by Year", xlabel="Year", ylabel="Proportion of Question Type")
plot!(p,2008:2023, avg_diff[1, 1:16], label="Medium", ribbon=std_diff[1, :], lw=2, m=:o, ms=3, ls=:dot, color=:blue)
plot!(p,2008:2023, avg_diff[2, 1:16], label="Hard", ribbon=std_diff[2, :], lw=2, m=:square, ls=:dash, ms=3, color=:red)
plot!(p,2008:2023, avg_diff[3, 1:16], label="Easy", ribbon=std_diff[3, :], lw=2, m=:x, ms=3, color=:green)
vline!(p,[2014], label=false, color=:black, lw=2, ls=:dash)
annotate!(p,(2011 + 2013) / 2, 0.5, text(" ← True Classes", :black, :center, 10))
annotate!(p,(2017 + 2018) / 2, 0.2, text("Predicted Classes →", :black, :center, 10))
savefig(p , "../results/fig5b.png")