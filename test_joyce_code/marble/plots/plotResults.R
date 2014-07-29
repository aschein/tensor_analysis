library(rjson)
library(plyr)
library(reshape2)
library(xtable)

iterResults <- fromJSON(file="iterresults.json")
itBasics <- ldply(iterResults, function(x) {
	x$fms <- NULL
	return(data.frame(x))
})
iterBasics <- subset(itBasics, iters != 2)
iterBasics$ll <- -iterBasics$ll
iterBasics <- melt(iterBasics, id.vars=c("iters"))
iterBasics <- subset(iterBasics, variable %in% c("comp", "totalMult", "ll"))
iterBasics$variable <- factor(iterBasics$variable, labels=c("Multiplicative Updates", "Time (secs)", "Log Likelihood"))
mir.stats <- ddply(iterBasics, .(iters, variable), summarise, median=median(value), avg=mean(value))
it.fms <- ldply(iterResults, function(x) {
	fms <- x$fms
	fms.df <- NULL
	for (mode in 1:3) {
		fms.df <- rbind(fms.df, cbind(data.frame(mode=mode-1, value =fms[[as.character(mode-1)]])))
	}
	return(cbind(expt=x$expt, iters= x$iters, fms.df))
})
it.fms <- subset(it.fms, iters != 2)
tmp <- ddply(it.fms, .(iters, mode), summarise, median=median(value), avg=mean(value))
tmp$variable <- paste("Mode", tmp$mode, "score", sep=" ")
tmp$mode <- NULL
itr <- rbind(mir.stats, tmp)
itr$iters <- as.integer(itr$iters)

## clean up the variable names
colnames(itr) <- c("Iterations", "Measurement", "Median", "Mean")
itr<- itr[order(itr$Measurement), ]
print(xtable(itr, digits=3), include.rownames=FALSE)

## Function to calculate the mode statistics based on the frame
calculateModeStats <- function(projFrame, varName) {
	 tmp <- ddply(projFrame, .(proj, mode), summarise, median=median(value), avg=mean(value))
	 tmp$variable <- paste("Mode", tmp$mode, varName, sep=" ")
	 tmp$mode <- NULL
	 return(tmp)
}

## analyze projection results
projResults <- fromJSON(file="projresults.json")
pr.basics <- ldply(projResults, function(x) {
	x$nonzero <- NULL
	x$fms <- NULL
	return(data.frame(x))
})
pr.basics <- subset(pr.basics, select=-c(totalOuter, totalMult))
pr.basics$ll <- -pr.basics$ll
mode.cols <- grep("Mode", colnames(pr.basics))
m.basics <- melt(pr.basics[, -mode.cols], id=c("proj", "expt"))
pr.sum <- ddply(m.basics, .(proj, variable), summarise, median=median(value), avg=mean(value))
levels(pr.sum$variable) <- c("Log Likelihood", "Time (secs)")

pr.nnz <- ldply(projResults, function(x) {
	nnz <- x$nonzero
	nnz.df <- NULL
	for (mode in 1:length(nnz)) {
		nnz.df <- rbind(nnz.df, cbind(data.frame(mode=mode-1, value =nnz[[mode]])))
	}
	return(cbind(expt=x$expt, proj= x$proj, nnz.df))
})
nnz.sum <- calculateModeStats(pr.nnz, "NNZ")

pr.fms <- ldply(projResults, function(x) {
	fms <- x$fms
	fms.df <- NULL
	for (mode in 1:3) {
		fms.df <- rbind(fms.df, cbind(data.frame(mode=mode-1, value =fms[[as.character(mode-1)]])))
	}
	return(cbind(expt=x$expt, proj= x$proj, fms.df))
})
fms.sum <- calculateModeStats(pr.fms, "score")

## bind them all together
presults <- rbind(pr.sum, nnz.sum, fms.sum)
presults <- subset(presults, proj %in% c("none", "full", "gamma"))
presults$proj <- factor(presults$proj, labels=c("None", "Full", "Gradual"))
presults <- presults[order(presults$variable), ]
colnames(presults) <- c("Projection", "Measurement", "Median", "Mean")
print(xtable(presults, digits=3), include.rownames=FALSE)

## Function to calculate the mode statistics based on the frame
calculateSimModeStats <- function(dataFr, varName) {
	 tmp <- ddply(dataFr, .(type, mode), summarise, median=median(value), avg=mean(value))
	 tmp$variable <- paste("Mode", tmp$mode, varName, sep=" ")
	 tmp$mode <- NULL
	 return(tmp)
}

simResults <- fromJSON(file="simresults.json")
sim.comp <- ldply(simResults, function(x) {
	return(data.frame(expt=x$sample, type= x$type, value=x$comp))
})
sim.comp <- subset(sim.comp, type != "Limestone")
sim.comp <- ddply(sim.comp, .(type), summarise, median=median(value), avg=mean(value))
print(xtable(sim.comp, digits=2), include.rownames=FALSE)

sim.nnz <- ldply(simResults, function(x) {
	nnz <- x$nonzero
	nnz.df <- NULL 
	for (mode in 1:length(nnz)) {
		nnz.df <- rbind(nnz.df, cbind(data.frame(mode=mode-1, value =nnz[[mode]])))
	}
	return(cbind(expt=x$sample, type= x$type, nnz.df))
})
sim.nnz <- subset(sim.nnz, type != "Limestone")
nnz.sum <- calculateSimModeStats(sim.nnz, "NNZ")
print(xtable(nnz.sum, digits=3), include.rownames=FALSE)


sim.fms <- ldply(simResults, function(x) {
	fms <- x$fms
	fms.df <- NULL
	for (mode in 1:3) {
		fms.df <- rbind(fms.df, cbind(data.frame(mode=mode-1, value =fms[[as.character(mode-1)]])))
	}
	return(cbind(expt=x$sample, type= x$type, fms.df))
})
sim.fms <- subset(sim.fms, type != "Limestone")
fms.sum <- calculateSimModeStats(sim.fms, "FMS")
print(xtable(nnz.sum, digits=3), include.rownames=FALSE)
