# install.packages("RcppCNPy")
# install.packages("PRROC")
# source("http://bioconductor.org/biocLite.R")
# biocLite("rhdf5")
# library("RcppCNPy")
library("rhdf5")
library("PRROC")

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "valid"
}

expname <- args[1]
dataset <- args[2]
resdir <- Sys.getenv("RESULT_DIR")
scores <- h5read(file=sprintf("%s/predictions-best/%s-%s_full.h5", resdir, expname, dataset), name="preds")
print(dim(scores))
scores <- t(scores)
labels <- h5read(file="data/processed_data/labels_full.h5", name=dataset)
# matrix with 919 rows and 8000 cols
calcfn <- function(labels, scores, mc_cores = getOption("mc.cores", 2L)) {
  calc_col <- function(c) {
    PRROC::pr.curve(scores.class0 = scores[, c], weights.class0 = labels[, c])$auc.davis.goadrich
  }
  parallel::mcmapply(calc_col, 1:ncol(labels), SIMPLIFY = TRUE)
}

labels <- labels[,c(1:(ncol(labels)%/%2))]
labels <- t(labels)

mc_cores <- parallel::detectCores(all.tests = FALSE, logical = TRUE)
print(mc_cores)
print(dim(labels))
print(dim(scores))
aup <- calcfn(labels, scores, mc_cores)
# ma <- matrix(aup)
# da <- as.data.frame(ma)
# colnames(da) <- c("adjlkkl")
write.csv(aup, file=sprintf("%s/auprcs/%s-%s.csv", resdir, expname, dataset))
# now to csv it