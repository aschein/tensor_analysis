library(rjson)
library(plyr)
library(reshape2)
library(ggplot2)
library(xtable)

## first plot the cp-apr histogram
cprHist <- fromJSON(file="cpapr-hist.json")
countHist <- ldply(cprHist, function(x) {
	histFrame = data.frame(count=x$count, value=x$bins[-11])
	return(cbind(expt=x$expt, mode=x$mode, histFrame))
})

sum.count <- ddply(countHist, .(mode, value), summarise, count=median(count))
sum.color = c("#77933C","#4F81BD","#C0504D")
for (n in 0:2) {
	ggplot(subset(sum.count, mode == n), aes(x=value, y=count)) + theme_bw() +
		geom_bar(stat = "identity", fill=sum.color[n+1]) +  scale_y_log10("Count") + 
		xlab("Non-zero value")
	ggsave(file=paste("cpapr-hist-", n, ".pdf", sep=""), width=4, height=4)
}

base.df <- fromJSON(file="../results/baseline-results.json")
base.df <- ldply(base.df, function(x) {
	return(data.frame(type="Baseline", seed=x$seed, auc=x$auc))
})
base.df <- ddply(base.df, .(type), summarise, avg=mean(auc), sd=sd(auc), n=length(auc))
calculateCI <- function(df) {
	df$upper <- df$avg + (qt(0.975, df=df$n) * df$sd / sqrt(df$n))
	df$lower <- df$avg - (qt(0.975, df=df$n) * df$sd / sqrt(df$n))
	return(df)
}
base.df <- calculateCI(base.df)

auc.df <- fromJSON(file="cms-results.json")
auc.df <- ldply(auc.df, function(x) {
	tmp <- data.frame(x$auc)
	return(cbind(expt=x$expt, R = x$R, tmp))
})
m.auc <- melt(auc.df, id=c("expt", "R"), variable="type", value.name="auc")
m.auc <- subset(m.auc, type != 'Baseline', select=-c(expt))
m.auc$Model <- factor(m.auc$type, labels=c("Marble", "CP-APR"))
auc.df <- ddply(m.auc, .(R, Model), summarise, avg=mean(auc), sd=sd(auc), n=length(auc))
auc.df <- calculateCI(auc.df)

dw = 1
ggplot(auc.df, aes(x=R, y=avg, colour=Model, shape=Model)) + 
		theme_bw() + geom_point(size=3, position=position_dodge(width=dw)) + scale_colour_brewer(palette="Dark2") + 
		geom_line() + geom_errorbar(aes(ymin=lower, ymax=upper), position=position_dodge(width=dw)) + 
		geom_hline(yintercept=base.df$avg, colour="#535353") + scale_shape_manual(values=c(17, 15)) + 
		annotate("text", 10, (base.df$avg-0.005), label="Baseline", vjust=0, size=4) + 
		geom_hline(yintercept=base.df$lower, colour="#535353", linetype="dashed") +  
		geom_hline(yintercept=base.df$upper, colour="#535353", linetype="dashed") +  
		scale_y_continuous("AUC") + xlab("Number of Phenotypes") + theme(legend.position=c(0.87, 0.20))
ggsave(file="AUC.pdf", width=6.5, height=3.5)

bias.df <- fromJSON(file="biasOut.json")
diag <- bias.df[[1]]
diag.df <- ldply(diag, print)
colnames(diag.df) <- c("Diagnosis", "Value")
print(xtable(diag.df, digits=3), include.rownames=FALSE)
proc <- bias.df[[2]]
proc.df <- ldply(proc, print)
colnames(proc.df) <- c("Procedure", "Value")
print(xtable(proc.df, digits=3), include.rownames=FALSE)

comp.df <- fromJSON(file="comp-cms-results.json")
comp.df <- ldply(comp.df, function(x) {
	tmp <- data.frame(type=x$cat, value=x$values)
	tmp <- tmp[order(tmp$value, decreasing=TRUE), ]
	return(cbind(Model = x$Model, Mode = x$Mode, tmp))
})
## print only the marble ones first
print(xtable(subset(comp.df, Model == "Marble", select=c(type))), include.rownames=FALSE)
print(xtable(subset(comp.df, Model == "CPAPR", select=c(type))), include.rownames=FALSE)

chronic.df <- fromJSON(file="cms-chronic-results.json")
chronic.df <- ldply(chronic.df, function(x) {
	tmp <- rbind(data.frame(Mode=1, Value=x$Diagnosis),data.frame(Mode=2, Value=x$Procedure))
	return(cbind(Disease=x$Disease, R=x$R, tmp))
})

print(xtable(subset(chronic.df, Disease == "HF", select=c(Mode, R, Value))), include.rownames=FALSE)
print(xtable(subset(chronic.df, Disease == "Diabetes", select=c(Mode, R, Value))), include.rownames=FALSE)
print(xtable(subset(chronic.df, Disease == "Arthritis", select=c(Mode, R, Value))), include.rownames=FALSE)
