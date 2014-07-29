library(rjson)
library(plyr)
library(reshape2)
library(ggplot2)

## first plot the cp-apr histogram
cprHist <- fromJSON(file="bih-hist.json")
countHist <- ldply(cprHist, function(x) {
	histFrame = data.frame(count=x$count)
	return(cbind(expt=x$expt, mode=x$mode, thr=x$thr, histFrame))
})
ch <- subset(countHist, mode > 0)
ch$mode <- factor(ch$mode)
ch$thr <- factor(ch$thr)
ggplot(subset(ch, mode == 1), aes(x=thr, y=count, colour=mode)) + theme_bw() + geom_boxplot() + 
	scale_colour_manual(values=c("#3F6BAF"))+
	xlab("Threshold") + ylab("Non-zero Elements per Phenotype") + theme(legend.position="none")
ggsave(file="cpapr-thresh-1.pdf", width=5, height=4)
ggplot(subset(ch, mode == 2), aes(x=thr, y=count, colour=mode)) + theme_bw() + geom_boxplot() +
	scale_colour_manual(values=c("#BB3B3C")) + 
	xlab("Threshold") + ylab("Non-zero Elements per Phenotype") + theme(legend.position="none")
ggsave(file="cpapr-thresh-2.pdf", width=5, height=4)

chronic.df <- fromJSON(file="bih-pheno.json")
chronic.df <- ldply(chronic.df, function(x) {
	tmp <- rbind(data.frame(Mode=0, Value=x$Patient), 
		data.frame(Mode=1, Value=x$Diagnosis),
		data.frame(Mode=2, Value=x$Procedure))
	return(cbind(Disease=x$Disease, R=x$R, tmp))
})

library(xtable)
print(xtable(subset(chronic.df, Disease == "Hypertension", select=c(Mode, R, Value))), include.rownames=FALSE)
print(xtable(subset(chronic.df, Disease == "HF", select=c(Mode, R, Value))), include.rownames=FALSE)
print(xtable(subset(chronic.df, Disease == "Diabetes", select=c(Mode, R, Value))), include.rownames=FALSE)
print(xtable(subset(chronic.df, Disease == "Arthritis", select=c(Mode, R, Value))), include.rownames=FALSE)
