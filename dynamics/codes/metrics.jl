# %%
using RCall
@rlibrary ggplot2
using PyCall
# @pyimport pandas as pd
pd = pyimport("pandas")
using DataFrames
using CSV
using TableView
using DataFramesMeta
using ProgressMeter
using Statistics
#%%
# ROIS = DataFrame(
#   xmin = [20, 20, 20, 485, 960, 1435, 485, 960, 1435],
#   xmax = [320, 320, 320, 785, 1255, 1735, 785, 1255, 1735],
#   ymin = [315, 680, 1040, 505, 505, 505, 870, 870, 870],
#   ymax = [25, 385, 745, 205, 205, 205, 570, 570, 570],
#   type = ["P", "P", "P", "D", "D", "D", "D", "D", "D"],
# )

# ROIS = {
#     'P1': [(- 15, 280), (235, 30)],
#     'P2': [(- 15, 680), (235, 430)],
#     'P3': [(- 15, 1080), (235, 830)],
#     'A': [(+ 465, 520), (715, 270)],
#     'B': [(+ 955, 520), (1205, 270)],
#     'C': [(+1445, 520), (1695, 270)],
#     'D': [(+ 465, 870), (715, 630)],
#     'E': [(+ 955, 870), (1205, 630)],
#     'F': [(+1445, 870), (1695, 630)]
# }


ROIS = DataFrame(
  xmin = [-15, -15, -15, 465, 955, 1445, 465, 955, 1445],
  xmax = [235, 235, 235, 715, 1205, 1695, 715, 1205, 1695],
  ymax = [280, 680, 1080, 520, 520, 520, 870, 870, 870],
  ymin = [30, 430, 830, 270, 270, 270, 630, 630, 630],
  type = ["P", "P", "P", "D", "D", "D", "D", "D", "D"],
)




function plot_trial(sacc, fix, block)

  ROIS.xcenter = ROIS.xmin + (ROIS.xmax - ROIS.xmin) / 2
  ROIS.ycenter = ROIS.ymin + (ROIS.ymax - ROIS.ymin) / 2

  sacc_b1 = @where(sacc, :block .== block)
  fix_b1 = @where(fix, :block .== block)

  ggplot() +
  geom_segment(
    data = sacc_b1,
    aes(x = :sxp, y = :syp, xend = :exp, yend = :eyp),
    arrow = arrow(),
    size = 1,
    alpha = 0.5,
    color = "grey40",
  ) +
  geom_point(
    data = fix_b1,
    aes(x = :axp, y = :ayp, size = :dur),
    alpha = 0.5,
    color = "blue",
  ) +
  scale_x_continuous(expand = [0, 0], limits = [0, 1920]) +
  scale_y_reverse(expand = [0, 0], limits = [1080, 0]) +
  labs(x = "x-axis (pixels)", y = "y-axis (pixels)") +
  geom_rect(
    data = ROIS,
    mapping = aes(
      xmin = :xmin,
      xmax = :xmax,
      ymin = :ymin,
      ymax = :ymax,
      fill = :type,
    ),
    color = "black",
    alpha = 0.5,
  ) +
  geom_text(data=ROIS, aes(x=:xcenter, y=:ycenter, label=:type), size=12) +
  coord_fixed() # Keeps aspect ratio from getting distorted

  # ggsave("plot_trial.png")
end

function transpose_df(org_df)
  df = copy(org_df)
  colnames = names(df)
  df[!, :id] = 1:size(df, 1)
  dfl = stack(df, colnames)
  unstack(dfl, :variable, :id, :value)
end


function in_roi(x, y)
  is_in = Bool[]
  for row in Tables.namedtupleiterator(ROIS)
    if (row.xmin < x < row.xmax) & (row.ymax < y < row.ymin)
      append!(is_in, true)
    else
      append!(is_in, false)
    end
  end
  any(is_in)
end

function in_pr(x, y)
  op_rois = @where(ROIS, :type .=="P")
  is_in = Bool[]
  for row in Tables.namedtupleiterator(op_rois)
    if (row.xmin < x < row.xmax) & (row.ymax < y < row.ymin)
      append!(is_in, true)
    else
      append!(is_in, false)
    end
  end
  any(is_in)
end

function in_op(x, y)
  op_rois = @where(ROIS, :type .=="D")
  is_in = Bool[]
  # if x isa String
  #   print("X: $(x) Y:$(y)")
  #   x = parse(Int, split(x, '.')[1])
  #   y = parse(Int, split(y, '.')[1])
  # end

  for row in Tables.namedtupleiterator(op_rois)
    if (row.xmin < x < row.xmax) & (row.ymax < y < row.ymin)
      append!(is_in, true)
    else
      append!(is_in, false)
    end
  end
  any(is_in)
end

function in_item(beh, x, y, block, item)
  if block ∉ beh.idx
    return false
  end
  trial = @where(beh, :idx .== block).answers[1]
  item_pos = findall(x -> x == item, trial)[1] + 3
  roi = ROIS[item_pos, :]
  ((roi.xmin < x < roi.xmax) & (roi.ymax < y < roi.ymin))
end


cd("../")

# %%
participants = map(x -> split(x, ".")[1], readdir("asc"))
# participants = participants[1:2] # for debugging purposes
res = Any[]
@showprogress for partid in participants
  print(" $(partid)")

  part_result = Dict()
  beh = DataFrame!(CSV.File(("beh/$(partid)_beh.csv")))
  fix = DataFrame!(CSV.File("fix/$(partid)_fix.csv"))
  raw = DataFrame!(CSV.File("raw/$(partid)_raw.csv"))
  sacc = DataFrame!(CSV.File("sacc/$(partid)_sacc.csv"))

  # set experimental trials with rt > 10
  beh.idx = 1:nrow(beh)
  beh[!, "answers"] = map(row -> [parse(Int, x) for x in split(row[2:end-1], ",")], beh.answers)
  beh = @where(beh, :rt .> 10, :exp .== 1)
  raw = @where(raw, :ps .!= "NA")
  raw[!, "ps"] = [parse(Int,x) for x in raw.ps]
  sacc = @where(sacc, :sxp .!= "NA", :syp .!= "NA", :exp .!= "NA", :eyp .!= "NA")
  sacc.exp[1] isa String ? sacc[!, "exp"] = [parse(Float64,x) for x in sacc.exp] : ""
  sacc.eyp[1] isa String ? sacc[!, "eyp"] = [parse(Float64,x) for x in sacc.eyp] : ""
  sacc.sxp[1] isa String ? sacc[!, "sxp"] = [parse(Float64,x) for x in sacc.sxp] : ""
  sacc.syp[1] isa String ? sacc[!, "syp"] = [parse(Float64,x) for x in sacc.syp] : ""

  beh_v1 = @where(beh, :type .== "v1")
  beh_v2 = @where(beh, :type .== "v2")
  beh_v1_corr = @where(beh, :type .== "v1", :corr .== 1)
  beh_v2_corr = @where(beh, :type .== "v2", :corr .== 1)
  beh_v1_incorr = @where(beh, :type .== "v1", :corr .== 0)
  beh_v2_incorr = @where(beh, :type .== "v2", :corr .== 0)
  no_of_incorr_trials = size(@where(beh, :corr .== 0))[1]

  part_result["PART_ID"] = partid
  part_result["SEX"] = split(partid, "_")[end]
  part_result["AGE"] = split(partid, "_")[2]
  part_result["ACC"] = mean(beh.corr)
  part_result["ACC_V1"] = mean(beh_v1.corr)
  part_result["ACC_V2"] = mean(beh_v2.corr)

  V1_RS = Dict("D6" => 1, "D5" => 5/6, "D4" => 4/6, "D3" => 3/6, "D2"=> 2/6, "D1"=> 1/6)
  V2_RS = Dict("D6" => 1, "D5" => 4/6, "D4" => 4/6, "D3" => 2/6, "D2"=> 2/6, "D1"=> 0)
  part_result["V1_RS"] = mean(map(x -> V1_RS[x], beh_v1.choosed_option))
  part_result["V2_RS"] = mean(map(x -> V2_RS[x], beh_v2.choosed_option))
  # Total no of trials in v1 and v2 are equal, so above vals can be just averaged
  part_result["TRS"] = (part_result["V1_RS"] + part_result["V2_RS"]) / 2.0

  part_result["V1_ERS"] = mean(map(x -> V1_RS[x], beh_v1_incorr.choosed_option))
  part_result["V2_ERS"] = mean(map(x -> V2_RS[x], beh_v2_incorr.choosed_option))
  # errs may differ so must be wieghted
  part_result["ERS"] = (
  (part_result["V1_ERS"] * size(beh_v1_incorr)[1]) +
  (part_result["V2_ERS"] * size(beh_v2_incorr)[1])
  ) / no_of_incorr_trials

  part_result["LAT_P"] = mean(@where(beh, :corr .== 1).rt)
  part_result["LAT_N"] = mean(@where(beh, :corr .== 0).rt)
  part_result["LAT_V1_P"] = mean(@where(beh, :type .== "v1", :corr .== 1).rt)
  part_result["LAT_V1_N"] = mean(@where(beh, :type .== "v1", :corr .== 0).rt)
  part_result["LAT_V2_P"] = mean(@where(beh, :type .== "v2", :corr .== 1).rt)
  part_result["LAT_V2_N"] = mean(@where(beh, :type .== "v2", :corr .== 0).rt)

  part_result["PUP_SIZE_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, raw).ps)
  part_result["PUP_SIZE_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, raw).ps)
  part_result["PUP_SIZE_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, raw).ps)
  part_result["PUP_SIZE_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, raw).ps)
  part_result["PUP_SIZE_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, raw).ps)
  part_result["PUP_SIZE_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, raw).ps)

  sacc_ends_in_op = filter(row -> in_op(row.exp, row.eyp), sacc)
  sacc_ends_in_pr = filter(row -> in_pr(row.exp, row.eyp), sacc)
  sacc_ends_in_roi = filter(row -> in_roi(row.exp, row.eyp), sacc)

  part_result["NT_V1"] = size(filter(row -> row.block ∈ beh_v1.idx, sacc_ends_in_roi))[1] / size(beh_v1)[1]
  part_result["NT_V2"] = size(filter(row -> row.block ∈ beh_v2.idx, sacc_ends_in_roi))[1] / size(beh_v2)[1]
  part_result["NT_V1_P"] = size(filter(row -> row.block ∈ beh_v1_corr.idx, sacc_ends_in_roi))[1] / size(beh_v1_corr)[1]
  part_result["NT_V2_P"] = size(filter(row -> row.block ∈ beh_v2_corr.idx, sacc_ends_in_roi))[1] / size(beh_v2_corr)[1]
  part_result["NT_V1_N"] = size(filter(row -> row.block ∈ beh_v1_incorr.idx, sacc_ends_in_roi))[1] / size(beh_v1_incorr)[1]
  part_result["NT_V2_N"] = size(filter(row -> row.block ∈ beh_v2_incorr.idx, sacc_ends_in_roi))[1] / size(beh_v2_incorr)[1]

  part_result["NT_PR_V1"] = size(filter(row -> row.block ∈ beh_v1.idx, sacc_ends_in_pr))[1]  / size(beh_v1)[1]
  part_result["NT_PR_V2"] = size(filter(row -> row.block ∈ beh_v2.idx, sacc_ends_in_pr))[1] / size(beh_v2)[1]
  part_result["NT_PR_V1_P"] = size(filter(row -> row.block ∈ beh_v1_corr.idx, sacc_ends_in_pr))[1] / size(beh_v1_corr)[1]
  part_result["NT_PR_V2_P"] = size(filter(row -> row.block ∈ beh_v2_corr.idx, sacc_ends_in_pr))[1] / size(beh_v2_corr)[1]
  part_result["NT_PR_V1_N"] = size(filter(row -> row.block ∈ beh_v1_incorr.idx, sacc_ends_in_pr))[1] / size(beh_v1_incorr)[1]
  part_result["NT_PR_V2_N"] = size(filter(row -> row.block ∈ beh_v2_incorr.idx, sacc_ends_in_pr))[1] / size(beh_v2_incorr)[1]

  part_result["NT_OP_V1"] = size(filter(row -> row.block ∈ beh_v1.idx, sacc_ends_in_op))[1] / size(beh_v1)[1]
  part_result["NT_OP_V2"] = size(filter(row -> row.block ∈ beh_v2.idx, sacc_ends_in_op))[1] / size(beh_v2)[1]
  part_result["NT_OP_V1_P"] = size(filter(row -> row.block ∈ beh_v1_corr.idx, sacc_ends_in_op))[1] / size(beh_v1_corr)[1]
  part_result["NT_OP_V2_P"] = size(filter(row -> row.block ∈ beh_v2_corr.idx, sacc_ends_in_op))[1] / size(beh_v2_corr)[1]
  part_result["NT_OP_V1_N"] = size(filter(row -> row.block ∈ beh_v1_incorr.idx, sacc_ends_in_op))[1] / size(beh_v1_incorr)[1]
  part_result["NT_OP_V2_N"] = size(filter(row -> row.block ∈ beh_v2_incorr.idx, sacc_ends_in_op))[1] / size(beh_v2_incorr)[1]

  sacc_ends_in_corr = filter((row -> in_item(beh, row.exp, row.eyp, row.block, 6)), sacc)
  part_result["NT_CORR_V1"] = size(filter(row -> row.block ∈ beh_v1.idx, sacc_ends_in_corr))[1] / size(beh_v1)[1]
  part_result["NT_CORR_V2"] = size(filter(row -> row.block ∈ beh_v2.idx, sacc_ends_in_corr))[1] / size(beh_v2)[1]
  part_result["NT_CORR_V1_P"] = size(filter(row -> row.block ∈ beh_v1_corr.idx, sacc_ends_in_corr))[1] / size(beh_v1_corr)[1]
  part_result["NT_CORR_V2_P"] = size(filter(row -> row.block ∈ beh_v2_corr.idx, sacc_ends_in_corr))[1] / size(beh_v2_corr)[1]
  part_result["NT_CORR_V1_N"] = size(filter(row -> row.block ∈ beh_v1_incorr.idx, sacc_ends_in_corr))[1] / size(beh_v1_incorr)[1]
  part_result["NT_CORR_V2_N"] = size(filter(row -> row.block ∈ beh_v2_incorr.idx, sacc_ends_in_corr))[1] / size(beh_v2_incorr)[1]

  sacc_ends_in_err = vcat(filter((row -> in_item(beh, row.exp, row.eyp, row.block, 5)), sacc),
                          filter((row -> in_item(beh, row.exp, row.eyp, row.block, 4)), sacc),
                          filter((row -> in_item(beh, row.exp, row.eyp, row.block, 3)), sacc),
                          filter((row -> in_item(beh, row.exp, row.eyp, row.block, 2)), sacc),
                          filter((row -> in_item(beh, row.exp, row.eyp, row.block, 1)), sacc))
  part_result["NT_ERR_V1"] = size(filter(row -> row.block ∈ beh_v1.idx, sacc_ends_in_err))[1] / size(beh_v1)[1]
  part_result["NT_ERR_V2"] = size(filter(row -> row.block ∈ beh_v2.idx, sacc_ends_in_err))[1] / size(beh_v2)[1]
  part_result["NT_ERR_V1_P"] = size(filter(row -> row.block ∈ beh_v1_corr.idx, sacc_ends_in_err))[1] / size(beh_v1_corr)[1]
  part_result["NT_ERR_V2_P"] = size(filter(row -> row.block ∈ beh_v2_corr.idx, sacc_ends_in_err))[1] / size(beh_v2_corr)[1]
  part_result["NT_ERR_V1_N"] = size(filter(row -> row.block ∈ beh_v1_incorr.idx, sacc_ends_in_err))[1] / size(beh_v1_incorr)[1]
  part_result["NT_ERR_V2_N"] = size(filter(row -> row.block ∈ beh_v2_incorr.idx, sacc_ends_in_err))[1] / size(beh_v2_incorr)[1]

  fix_in_pr = filter(row -> in_pr(row.axp, row.ayp), fix)
  fix_dur_in_pr_by_block = combine(groupby(fix_in_pr, :block), :dur .=> sum)
  fix_dur_in_pr_by_block_v1 = filter(row -> row.block ∈ beh_v1.idx, fix_dur_in_pr_by_block)
  fix_dur_in_pr_by_block_v2 = filter(row -> row.block ∈ beh_v2.idx, fix_dur_in_pr_by_block)
  fix_dur_in_pr_by_block_v1_P = filter(row -> row.block ∈ beh_v1_corr.idx, fix_dur_in_pr_by_block)
  fix_dur_in_pr_by_block_v2_P = filter(row -> row.block ∈ beh_v2_corr.idx, fix_dur_in_pr_by_block)
  fix_dur_in_pr_by_block_v1_N = filter(row -> row.block ∈ beh_v1_incorr.idx, fix_dur_in_pr_by_block)
  fix_dur_in_pr_by_block_v2_N = filter(row -> row.block ∈ beh_v2_incorr.idx, fix_dur_in_pr_by_block)

  part_result["RT_PR_V1"] = mean(@transform(fix_dur_in_pr_by_block_v1, X = :dur_sum ./ (beh_v1.rt * 1000.0)).X)
  part_result["RT_PR_V2"] = mean(@transform(fix_dur_in_pr_by_block_v2, X = :dur_sum ./ (beh_v2.rt * 1000.0)).X)
  part_result["RT_PR_V1_P"] = mean(@transform(fix_dur_in_pr_by_block_v1_P, X = :dur_sum ./ (beh_v1_corr.rt * 1000.0)).X)
  part_result["RT_PR_V2_P"] = mean(@transform(fix_dur_in_pr_by_block_v2_P, X = :dur_sum ./ (beh_v2_corr.rt * 1000.0)).X)
  part_result["RT_PR_V1_N"] = mean(@transform(fix_dur_in_pr_by_block_v1_N, X = :dur_sum ./ (beh_v1_incorr.rt * 1000.0)).X)
  part_result["RT_PR_V2_N"] = mean(@transform(fix_dur_in_pr_by_block_v2_N, X = :dur_sum ./ (beh_v2_incorr.rt * 1000.0)).X)

  fix_in_op = filter(row -> in_op(row.axp, row.ayp), fix)
  fix_dur_in_op_by_block = combine(groupby(fix_in_op, :block), :dur .=> sum)
  fix_dur_in_op_by_block_v1 = filter(row -> row.block ∈ beh_v1.idx, fix_dur_in_op_by_block)
  fix_dur_in_op_by_block_v2 = filter(row -> row.block ∈ beh_v2.idx, fix_dur_in_op_by_block)
  fix_dur_in_op_by_block_v1_P = filter(row -> row.block ∈ beh_v1_corr.idx, fix_dur_in_op_by_block)
  fix_dur_in_op_by_block_v2_P = filter(row -> row.block ∈ beh_v2_corr.idx, fix_dur_in_op_by_block)
  fix_dur_in_op_by_block_v1_N = filter(row -> row.block ∈ beh_v1_incorr.idx, fix_dur_in_op_by_block)
  fix_dur_in_op_by_block_v2_N = filter(row -> row.block ∈ beh_v2_incorr.idx, fix_dur_in_op_by_block)

  part_result["RT_OP_V1"] = mean(@transform(fix_dur_in_op_by_block_v1, X = :dur_sum ./ (beh_v1.rt * 1000.0)).X)
  part_result["RT_OP_V2"] = mean(@transform(fix_dur_in_op_by_block_v2, X = :dur_sum ./ (beh_v2.rt * 1000.0)).X)
  part_result["RT_OP_V1_P"] = mean(@transform(fix_dur_in_op_by_block_v1_P, X = :dur_sum ./ (beh_v1_corr.rt * 1000.0)).X)
  part_result["RT_OP_V2_P"] = mean(@transform(fix_dur_in_op_by_block_v2_P, X = :dur_sum ./ (beh_v2_corr.rt * 1000.0)).X)
  part_result["RT_OP_V1_N"] = mean(@transform(fix_dur_in_op_by_block_v1_N, X = :dur_sum ./ (beh_v1_incorr.rt * 1000.0)).X)
  part_result["RT_OP_V2_N"] = mean(@transform(fix_dur_in_op_by_block_v2_N, X = :dur_sum ./ (beh_v2_incorr.rt * 1000.0)).X)

  fix_in_corr = filter((row -> in_item(beh, row.axp, row.ayp, row.block, 6)), fix)
  fix_dur_in_corr_by_block = combine(groupby(fix_in_corr, :block), :dur .=> sum)
  fix_dur_in_corr_by_block_v1 = filter(row -> row.block ∈ beh_v1.idx, fix_dur_in_corr_by_block)
  fix_dur_in_corr_by_block_v2 = filter(row -> row.block ∈ beh_v2.idx, fix_dur_in_corr_by_block)
  fix_dur_in_corr_by_block_v1_P = filter(row -> row.block ∈ beh_v1_corr.idx, fix_dur_in_corr_by_block)
  fix_dur_in_corr_by_block_v2_P = filter(row -> row.block ∈ beh_v2_corr.idx, fix_dur_in_corr_by_block)
  fix_dur_in_corr_by_block_v1_N = filter(row -> row.block ∈ beh_v1_incorr.idx, fix_dur_in_corr_by_block)
  fix_dur_in_corr_by_block_v2_N = filter(row -> row.block ∈ beh_v2_incorr.idx, fix_dur_in_corr_by_block)

  part_result["RT_CORR_V1"] = mean(@transform(fix_dur_in_corr_by_block_v1, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_corr_by_block_v1.block, beh_v1).rt * 1000.0)).X)
  part_result["RT_CORR_V2"] = mean(@transform(fix_dur_in_corr_by_block_v2, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_corr_by_block_v2.block, beh_v2).rt * 1000.0)).X)
  part_result["RT_CORR_V1_P"] = mean(@transform(fix_dur_in_corr_by_block_v1_P, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_corr_by_block_v1_P.block, beh_v1_corr).rt * 1000.0)).X)
  part_result["RT_CORR_V2_P"] = mean(@transform(fix_dur_in_corr_by_block_v2_P, X = :dur_sum ./ (beh_v2_corr.rt * 1000.0)).X)
  part_result["RT_CORR_V1_N"] = mean(@transform(fix_dur_in_corr_by_block_v1_N, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_corr_by_block_v1_N.block, beh_v1_incorr).rt * 1000.0)).X)
  part_result["RT_CORR_V2_N"] = mean(@transform(fix_dur_in_corr_by_block_v2_N, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_corr_by_block_v2_N.block, beh_v2_incorr).rt * 1000.0)).X)

  fix_in_err = vcat(filter((row -> in_item(beh, row.axp, row.ayp, row.block, 5)), fix),
                    filter((row -> in_item(beh, row.axp, row.ayp, row.block, 4)), fix),
                    filter((row -> in_item(beh, row.axp, row.ayp, row.block, 3)), fix),
                    filter((row -> in_item(beh, row.axp, row.ayp, row.block, 2)), fix),
                    filter((row -> in_item(beh, row.axp, row.ayp, row.block, 1)), fix))

  fix_dur_in_err_by_block = combine(groupby(fix_in_err, :block), :dur .=> sum)
  fix_dur_in_err_by_block_v1 = filter(row -> row.block ∈ beh_v1.idx, fix_dur_in_err_by_block)
  fix_dur_in_err_by_block_v2 = filter(row -> row.block ∈ beh_v2.idx, fix_dur_in_err_by_block)
  fix_dur_in_err_by_block_v1_P = filter(row -> row.block ∈ beh_v1_corr.idx, fix_dur_in_err_by_block)
  fix_dur_in_err_by_block_v2_P = filter(row -> row.block ∈ beh_v2_corr.idx, fix_dur_in_err_by_block)
  fix_dur_in_err_by_block_v1_N = filter(row -> row.block ∈ beh_v1_incorr.idx, fix_dur_in_err_by_block)
  fix_dur_in_err_by_block_v2_N = filter(row -> row.block ∈ beh_v2_incorr.idx, fix_dur_in_err_by_block)

  part_result["RT_ERR_V1"] = mean(@transform(fix_dur_in_err_by_block_v1, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_err_by_block_v1.block, beh_v1).rt * 1000.0)).X)
  part_result["RT_ERR_V2"] = mean(@transform(fix_dur_in_err_by_block_v2, X = :dur_sum ./ (beh_v2.rt * 1000.0)).X)
  part_result["RT_ERR_V1_P"] = mean(@transform(fix_dur_in_err_by_block_v1_P, X = :dur_sum ./ (filter(row -> row.idx ∈ fix_dur_in_err_by_block_v1_P.block, beh_v1_corr).rt * 1000.0)).X)
  part_result["RT_ERR_V2_P"] = mean(@transform(fix_dur_in_err_by_block_v2_P, X = :dur_sum ./ (beh_v2_corr.rt * 1000.0)).X)
  part_result["RT_ERR_V1_N"] = mean(@transform(fix_dur_in_err_by_block_v1_N, X = :dur_sum ./ (beh_v1_incorr.rt * 1000.0)).X)
  part_result["RT_ERR_V2_N"] = mean(@transform(fix_dur_in_err_by_block_v2_N, X = :dur_sum ./ (beh_v2_incorr.rt * 1000.0)).X)

  mean_durr_in_pr_by_block = combine(groupby(fix_in_pr, :block), :dur .=> mean)
  part_result["DUR_PR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, mean_durr_in_pr_by_block).dur_mean)
  part_result["DUR_PR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, mean_durr_in_pr_by_block).dur_mean)
  part_result["DUR_PR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, mean_durr_in_pr_by_block).dur_mean)
  part_result["DUR_PR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, mean_durr_in_pr_by_block).dur_mean)
  part_result["DUR_PR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, mean_durr_in_pr_by_block).dur_mean)
  part_result["DUR_PR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, mean_durr_in_pr_by_block).dur_mean)

  mean_durr_in_op_by_block = combine(groupby(fix_in_op, :block), :dur .=> mean)
  part_result["DUR_OP_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, mean_durr_in_op_by_block).dur_mean)
  part_result["DUR_OP_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, mean_durr_in_op_by_block).dur_mean)
  part_result["DUR_OP_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, mean_durr_in_op_by_block).dur_mean)
  part_result["DUR_OP_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, mean_durr_in_op_by_block).dur_mean)
  part_result["DUR_OP_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, mean_durr_in_op_by_block).dur_mean)
  part_result["DUR_OP_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, mean_durr_in_op_by_block).dur_mean)

  mean_durr_in_corr_by_block = combine(groupby(fix_in_corr, :block), :dur .=> mean)
  part_result["DUR_CORR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, mean_durr_in_corr_by_block).dur_mean)
  part_result["DUR_CORR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, mean_durr_in_corr_by_block).dur_mean)
  part_result["DUR_CORR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, mean_durr_in_corr_by_block).dur_mean)
  part_result["DUR_CORR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, mean_durr_in_corr_by_block).dur_mean)
  part_result["DUR_CORR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, mean_durr_in_corr_by_block).dur_mean)
  part_result["DUR_CORR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, mean_durr_in_corr_by_block).dur_mean)

  mean_durr_in_err_by_block = combine(groupby(fix_in_err, :block), :dur .=> mean)
  part_result["DUR_ERR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, mean_durr_in_err_by_block).dur_mean)
  part_result["DUR_ERR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, mean_durr_in_err_by_block).dur_mean)
  part_result["DUR_ERR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, mean_durr_in_err_by_block).dur_mean)
  part_result["DUR_ERR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, mean_durr_in_err_by_block).dur_mean)
  part_result["DUR_ERR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, mean_durr_in_err_by_block).dur_mean)
  part_result["DUR_ERR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, mean_durr_in_err_by_block).dur_mean)

  sum_durr_in_pr_by_block = combine(groupby(fix_in_pr, :block), :dur .=> sum)
  part_result["FIX_PR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, sum_durr_in_pr_by_block).dur_sum)
  part_result["FIX_PR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, sum_durr_in_pr_by_block).dur_sum)
  part_result["FIX_PR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, sum_durr_in_pr_by_block).dur_sum)
  part_result["FIX_PR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, sum_durr_in_pr_by_block).dur_sum)
  part_result["FIX_PR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, sum_durr_in_pr_by_block).dur_sum)
  part_result["FIX_PR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, sum_durr_in_pr_by_block).dur_sum)

  sum_durr_in_op_by_block = combine(groupby(fix_in_op, :block), :dur .=> sum)
  part_result["FIX_OP_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, sum_durr_in_op_by_block).dur_sum)
  part_result["FIX_OP_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, sum_durr_in_op_by_block).dur_sum)
  part_result["FIX_OP_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, sum_durr_in_op_by_block).dur_sum)
  part_result["FIX_OP_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, sum_durr_in_op_by_block).dur_sum)
  part_result["FIX_OP_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, sum_durr_in_op_by_block).dur_sum)
  part_result["FIX_OP_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, sum_durr_in_op_by_block).dur_sum)

  sum_durr_in_corr_by_block = combine(groupby(fix_in_corr, :block), :dur .=> sum)
  part_result["FIX_CORR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, sum_durr_in_corr_by_block).dur_sum)
  part_result["FIX_CORR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, sum_durr_in_corr_by_block).dur_sum)
  part_result["FIX_CORR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, sum_durr_in_corr_by_block).dur_sum)
  part_result["FIX_CORR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, sum_durr_in_corr_by_block).dur_sum)
  part_result["FIX_CORR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, sum_durr_in_corr_by_block).dur_sum)
  part_result["FIX_CORR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, sum_durr_in_corr_by_block).dur_sum)

  sum_durr_in_err_by_block = combine(groupby(fix_in_err, :block), :dur .=> sum)
  part_result["FIX_ERR_V1"] = mean(filter(row -> row.block ∈ beh_v1.idx, sum_durr_in_err_by_block).dur_sum)
  part_result["FIX_ERR_V2"] = mean(filter(row -> row.block ∈ beh_v2.idx, sum_durr_in_err_by_block).dur_sum)
  part_result["FIX_ERR_V1_P"] = mean(filter(row -> row.block ∈ beh_v1_corr.idx, sum_durr_in_err_by_block).dur_sum)
  part_result["FIX_ERR_V2_P"] = mean(filter(row -> row.block ∈ beh_v2_corr.idx, sum_durr_in_err_by_block).dur_sum)
  part_result["FIX_ERR_V1_N"] = mean(filter(row -> row.block ∈ beh_v1_incorr.idx, sum_durr_in_err_by_block).dur_sum)
  part_result["FIX_ERR_V2_N"] = mean(filter(row -> row.block ∈ beh_v2_incorr.idx, sum_durr_in_err_by_block).dur_sum)
  append!(res, [part_result])

end
res = pd.DataFrame(res)
# cols = py"['PART_ID', 'SEX', 'AGE'] + sorted(list(set(res.columns) - set(['PART_ID', 'SEX', 'AGE'])))"
# cols = res.columns
# filter!(e -> e ∉["PART_ID", "SEX", "AGE"], cols)
# cols = vcat(["PART_ID", "SEX", "AGE"], sort(cols))
# res = res[cols]
res.to_csv("result.csv", header=true, float_format="%.4f")
