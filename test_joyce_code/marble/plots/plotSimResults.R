library(rjson)
library(plyr)
library(reshape2)
library(xtable)
library(ggplot2)

### Function to extract FMS data from the simulation results
extractFMSData <- function(lresults, varName) {
	return(
		ldply(lresults, function(x) {
			field <- x[['fms']]
			field.df <- NULL
			for (mode in 0:2) {
				field.df <- rbind(field.df, cbind(data.frame(mode=mode, value = field[[as.character(mode)]])))
			}
			return(cbind(expt=x$expt, type= x[[varName]], field.df))
		})
	)
}

##########		ITERATION RESULTS 		##########
iterResults <- fromJSON(file="iterresults.json")
it.fms <- extractFMSData(iterResults, "iters")
it.fms <- subset(it.fms, type != 2)
it.fms <- ddply(it.fms, .(type, mode), summarise, median=median(value), avg=mean(value), sd=sd(value), n=length(value))
tmp$Iterations <- factor(tmp$type)
tmp$Mode <- tmp$mode + 1
ggplot(tmp, aes(x=Mode, y=avg, color=Iterations, shape=Iterations)) + geom_line() + geom_point(size=5) + 
 	theme_bw() + ylim(0.7, 0.8) + scale_x_continuous(breaks=c(1,2,3)) + 
	scale_colour_brewer(palette="Dark2") + ylab("Similarity") + theme(legend.position="top")
ggsave(file="iteration.pdf", width=5, height=4)

## Function to calculate the mode statistics based on the frame
calculateModeStats <- function(projFrame, varName) {
	 tmp <- ddply(projFrame, .(proj, mode), summarise, median=median(value), avg=mean(value))
	 tmp$variable <- tmp$mode
	 tmp$mode <- NULL
	 return(tmp)
}

##########		PROJECTION RESULTS 		##########
projResults <- fromJSON(file="projresults.json")
proj.fms <- extractFMSData(projResults, "proj")
proj.fms <- subset(proj.fms, type %in% c("none", "full", "gamma"))
proj.fms <- ddply(proj.fms, .(type, mode), summarise, median=median(value), avg=mean(value))

pr.nnz <- ldply(projResults, function(x) {
	nnz <- x$nonzero
	nnz.df <- NULL
	for (mode in 1:length(nnz)) {
		nnz.df <- rbind(nnz.df, cbind(data.frame(mode=mode-1, value =nnz[[mode]])))
	}
	return(cbind(expt=x$expt, proj= x$proj, nnz.df))
})
nnz.sum <- calculateModeStats(pr.nnz, "NNZ")
proj.df <- subset(nnz.sum, proj %in% c("none", "full", "gamma"))
proj.df$NNZ <- proj.df$avg
proj.df$NZR[proj.df$Mode == 0] <- proj.df$NNZ[proj.df$Mode == 0] / 20
proj.df$NZR[proj.df$Mode == 1] <- proj.df$NNZ[proj.df$Mode == 1] / 15
proj.df$NZR[proj.df$Mode == 2] <- proj.df$NNZ[proj.df$Mode == 2] / 10
proj.df$Score <- proj.fms$avg

proj.df$Projection <- factor(proj.df$proj, labels=c("None", "Full", "Gradual"))
proj.df$Mode <- proj.df$variable + 1
proj.df$Mode <- factor(proj.df$Mode, labels=c("Mode 1","Mode 2","Mode 3"))

ggplot(proj.df, aes(x=NZR, y=Score, color=Projection)) + geom_point(size=5) + theme_bw() + 
	scale_color_brewer(palette="Dark2") + ylim(0.85, 0.95) + ylab("Similarity") + 
	theme(legend.position="top") + xlab("Non-zero Ratio") + facet_grid(.~Mode, scales="free") + 
	geom_vline(xintercept=1, linetype="longdash")
ggsave(file="projection.pdf", width=5, height=4)


##########		COMPARISON BETWEEN CP-APR, LIMESTONE, MARBLE 		##########
simResults <- fromJSON(file="simresults.json")
sim.nnz <- extractData(simResults, "nonzero")
sim.nnz <- ldply(simResults, function(x) {
	nnz <- x$nonzero
	nnz.df <- NULL 
	for (mode in 1:length(nnz)) {
		nnz.df <- rbind(nnz.df, cbind(data.frame(mode=mode-1, value =nnz[[mode]])))
	}
	return(cbind(expt=x$sample, type= x$type, nnz.df))
})
# sim.nnz <- subset(sim.nnz, type != "Limestone")
sim.nnz <- ddply(sim.nnz, .(type, mode), summarise, median=median(value), avg=mean(value))

sim.fms <- extractFMSData(simResults, "type")
# sim.fms <- subset(sim.fms, type != "Limestone")
sim.fms <- ddply(sim.fms, .(type, mode), summarise, median=median(value), avg=mean(value))
sim.nnz <- sim.nnz[order(sim.nnz$type, sim.nnz$mode),-3]
sim.fms<- sim.fms[order(sim.fms$type, sim.fms$mode), ]
sim.df <-cbind(sim.nnz, sim.fms$avg)
colnames(sim.df) <- c("Model", "Mode", "NNZ", "Score")
sim.df$NZR[sim.df$Mode == 0] <- sim.df$NNZ[sim.df$Mode == 0]/20
sim.df$NZR[sim.df$Mode == 1] <- sim.df$NNZ[sim.df$Mode == 1]/10
sim.df$NZR[sim.df$Mode == 2] <- sim.df$NNZ[sim.df$Mode == 2]/5
sim.df$Mode <- factor(sim.df$Mode, labels=c("Mode 1","Mode 2","Mode 3"))
sim.df$Model <- factor(sim.df$Model, labels=c("Marble", "CP-APR", "HT CP-APR"))

ggplot(sim.df, aes(x=NZR, y=Score, color=Model)) + geom_vline(xintercept=1, linetype="longdash") + 
	geom_point(size=5) + theme_bw() + ylim(0.8, 0.85) + ylab("Similarity") +
	scale_color_brewer(palette="Dark2") + facet_grid(.~Mode, scales="free") + 
	theme(legend.position="top") + xlab("Non-zero Ratio") 
ggsave(file="sim-model.pdf", width=5, height=4)
