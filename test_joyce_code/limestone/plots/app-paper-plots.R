require(ggplot2)
require(reshape2)
require(plyr)
require(scales)
require(ROCR)
library(xtable)

##### Rank versus log likelihood ##############
#rank.df <- read.csv('uniq-rank.csv', sep='|', header=FALSE)
#colnames(rank.df) <- c("Model", "Initial", "Rank", "Iterations", "LS", "LL")
#rank.df$Initial <- factor(rank.df$Initial)
#rank.df <- ddply(rank.df, .(Model, Rank), summarise, avg=-mean(LL), sd=sd(LL), n=length(LL))
#rank.df$upper <- rank.df$avg + (qt(0.975, df=rank.df$n) * rank.df$sd / sqrt(rank.df$n))
#rank.df$lower <- rank.df$avg - (qt(0.975, df=rank.df$n) * rank.df$sd / sqrt(rank.df$n))
#
#ggplot(rank.df, aes(x=Rank, y=avg)) + theme_bw() + geom_point() + geom_line(colour="#0571B0") + 
#		geom_errorbar(aes(ymin=lower, ymax=upper, alpha=0.4), width=7) + scale_colour_brewer(palette="Paired") + 
#		scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
#				labels = trans_format("log10", 
#						math_format(.x, format=function(x){format(x, nsmall=2)}))) + 
#		xlab("Number of Phenotypes")+ 
#		ylab(expression(paste("Negative Log Likelihood (", 10^x, ")", sep=""))) + 
#		theme(legend.position="none")
#ggsave(file="rank-ll.pdf", width=6, height=4)
#
###### Rank versus Scores ##############
#scores.df <- read.csv('uniq-rank-scores.csv', sep='|', header=FALSE)
#colnames(scores.df) <- c("Model", "Type", "Method", "A", "B", "Mode", "FactorA", "FactorB", "Value")
#scores.df <- ddply(scores.df, .(Model), function(x) {
#			rank = subset(rank.df, Model == x$Model[1])$Rank[1]
#			x$Rank <- rank;
#			return(x)
#		});
#scores.df <- subset(scores.df, Mode >= 0) # Ignore the lambda modes
#scores.df <- ddply(scores.df, .(Rank, Type, Method, Mode), summarise, avg=mean(Value), sd=sd(Value), n=length(Value))
#scores.df$upper <- scores.df$avg + (qt(0.975, df=scores.df$n) * scores.df$sd / sqrt(scores.df$n))
#scores.df$lower <- scores.df$avg - (qt(0.975, df=scores.df$n) * scores.df$sd / sqrt(scores.df$n))
#scores.df$upper[scores.df$upper > 1] <- 1
#scores.df$lower[scores.df$lower < 0] <- 0
#scores.df$Method <- factor(scores.df$Method, labels=c("Greedy", "Top10"))
#scores.df$Type <- factor(scores.df$Type, labels=c("FMS", "FOS"))
#scores.df$Mode <- factor(scores.df$Mode, labels=c("Patient", "Diagnosis", "Medication"))
#ggplot(subset(scores.df, Method=="Greedy" & Type == "FMS"), aes(x=Rank, y=avg, colour=Mode)) + 
#		theme_bw() + geom_point() + geom_line() + geom_errorbar(aes(ymin=lower, ymax=upper), width=5) + 
#		scale_colour_manual(values=c("#4DAF4A","#377EB8","#E41A1C")) + 
#		scale_y_continuous("Score", limits=c(0,1)) + xlab("Number of Phenotypes") + 
#		theme(legend.position=c(0.85,0.17))
#ggsave(file="rank-score.pdf", width=6, height=4)
#
#ggplot(subset(scores.df, Method=="Greedy" & Type == "FMS"), aes(x=Rank, y=avg, colour=Mode)) + 
#		theme_bw() + geom_point() + geom_line() + geom_errorbar(aes(ymin=lower, ymax=upper), width=5) + 
#		scale_colour_brewer(palette="Set1") + scale_y_continuous("Cosine Similarity", limits=c(0,1)) + xlab("Number of Phenotypes") + 
#		theme(legend.position='top')
#ggsave(file="fms-presentation.pdf", width=6, height=4)

##### Iteration versus log likelihood fit ##############
rm(list=ls())
iter.df <- read.csv('uniq-results.csv', sep='|', header=FALSE)
colnames(iter.df) <- c("Model", "Initial", "Rank", "Iterations", "LS", "LL")
tmp.df <- iter.df
iter.df <- ddply(iter.df, .(Iterations), summarise, avg=-mean(LL), sd=sd(LL), n=length(LL))
iter.df$upper <- iter.df$avg + (qt(0.975, df=iter.df$n) * iter.df$sd / sqrt(iter.df$n))
iter.df$lower <- iter.df$avg - (qt(0.975, df=iter.df$n) * iter.df$sd / sqrt(iter.df$n))
ggplot(iter.df, aes(x=Iterations, y=avg)) + theme_bw() + geom_point(size=1.25) + geom_line() + 
    geom_errorbar(aes(ymin=lower, ymax=upper, alpha=0.8, width=7)) +
		scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
				labels = trans_format("log10", 
						math_format(.x, format=function(x){format(x, nsmall=2)}))) +
		theme(legend.position="none") + xlab("Iterations") + ylab(expression(paste("Negative Log-likelihood (", 10^x, ")", sep="")))
ggsave(file="iter-ll.pdf", width=4.5, height=3.38)

##### Iteration versus Scores ##############
scores.df <- read.csv('uniq-results-scores.csv', sep='|', header=FALSE)
colnames(scores.df) <- c("Model", "Type", "Method", "A", "B", "Mode", "FactorA", "FactorB", "Value")
scores.df <- ddply(scores.df, .(Model), function(x) {
			iteration = subset(tmp.df, Model == x$Model[1])$Iterations[1]
			x$Iterations <- iteration;
			return(x)
		});
scores.df <- subset(scores.df, Mode >= 0) # Ignore the lambda modes
scores.df <- ddply(scores.df, .(Iterations, Type, Method, Mode), summarise, avg=mean(Value), sd=sd(Value), n=length(Value))
scores.df$upper <- scores.df$avg + (qt(0.975, df=scores.df$n) * scores.df$sd / sqrt(scores.df$n))
scores.df$lower <- scores.df$avg - (qt(0.975, df=scores.df$n) * scores.df$sd / sqrt(scores.df$n))
scores.df$upper[scores.df$upper > 1] <- 1
scores.df$lower[scores.df$lower < 0] <- 0
scores.df$Method <- factor(scores.df$Method, labels=c("Greedy", "Top10"))
scores.df$Type <- factor(scores.df$Type, labels=c("FMS", "FOS"))
scores.df$Mode <- factor(scores.df$Mode, labels=c("Patient", "Diagnosis", "Medication"))
ggplot(subset(scores.df, Method=="Greedy" & Type == "FMS"), aes(x=Iterations, y=avg, colour=Mode, shape=Mode)) + 
		theme_bw() + geom_point() + geom_line(aes(linetype=Mode)) + geom_errorbar(aes(ymin=lower, ymax=upper), width=5) + 
		scale_colour_manual(values=c("#4DAF4A","#377EB8","#E41A1C")) + 
		scale_linetype_manual(values=c("dotted", "solid", "longdash")) + 
		scale_y_continuous("Similarity", limits=c(0.5,0.9)) + xlab("Iterations") + 
		theme(legend.position=c(0.83,0.2))
ggsave(file="iter-score.pdf", width=4.5, height=3.38)

##### Perturbation versus Scores ##############
rm(list=ls())
df <- read.csv('perturb-expt2.csv', header=FALSE)
colnames(df) <-c("expt", "noise", "mode", "value")
sum.df <- ddply(df, .(noise, mode), summarise, avg=mean(value), sd=sd(value), n=length(value))
sum.df$noise <- sum.df$noise * 100
sum.df$upper <- sum.df$avg + (qt(0.975, df=sum.df$n) * sum.df$sd / sqrt(sum.df$n))
sum.df$lower <- sum.df$avg - (qt(0.975, df=sum.df$n) * sum.df$sd / sqrt(sum.df$n))
sum.df$mode <- factor(sum.df$mode, labels=c("Patient", "Diagnosis", "Medication"))

ggplot(sum.df, aes(x=noise, y=avg, colour=mode, shape=mode)) + theme_bw() + 
		geom_point(position=position_dodge(width=1)) + 	geom_line(aes(linetype=mode)) + 
		geom_errorbar(aes(ymin=lower, ymax=upper), width=2, position=position_dodge(width=1)) + 
		scale_colour_manual(values=c("#4DAF4A","#377EB8","#E41A1C")) + 
		scale_linetype_manual(values=c("dotted", "solid", "longdash")) + 
		scale_y_continuous("Similarity", limits=c(0.5,1)) +
		xlab("Perturbation Percentage") + theme(legend.position=c(0.83,0.2))
ggsave(file="perturb2-score.pdf", width=4.5, height=3.38)

df <- read.csv('perturb-expt1.csv', header=FALSE)
colnames(df) <-c("expt", "noise", "mode", "value")
sum.df <- ddply(df, .(noise, mode), summarise, avg=mean(value), sd=sd(value), n=length(value))
sum.df$noise <- sum.df$noise * 100
sum.df$upper <- sum.df$avg + (qt(0.975, df=sum.df$n) * sum.df$sd / sqrt(sum.df$n))
sum.df$lower <- sum.df$avg - (qt(0.975, df=sum.df$n) * sum.df$sd / sqrt(sum.df$n))
sum.df$mode <- factor(sum.df$mode, labels=c("Patient", "Diagnosis", "Medication"))

ggplot(sum.df, aes(x=noise, y=avg, colour=mode, shape=mode)) + theme_bw() + 
		geom_point(position=position_dodge(width=1)) + 	geom_line(aes(linetype=mode)) + 
		geom_errorbar(aes(ymin=lower, ymax=upper), width=2, position=position_dodge(width=1)) + 
		scale_colour_manual(values=c("#4DAF4A","#377EB8","#E41A1C")) + 
		scale_linetype_manual(values=c("dotted", "solid", "longdash")) + 
		scale_y_continuous("Similarity", limits=c(0.5,1)) +
		xlab("Perturbation Percentage") + theme(legend.position=c(0.83,0.2))
ggsave(file="perturb1-score.pdf", width=4.5, height=3.38)


########## Patients vs CPU Time #################
rm(list=ls())
pat.cpu <- read.csv('patient-results.csv', header=FALSE)
colnames(pat.cpu) <- c("Patients", "Sample", "Method", "Time")
pat.cpu$Method <- factor(pat.cpu$Method, labels=c("Limestone", "NMF", "PCA"))
pat.df <- ddply(pat.cpu, .(Patients, Method), summarise, avg=mean(Time), sd=sd(Time), n=length(Time))
pat.df$upper <- pat.df$avg + (qt(0.975, df=pat.df$n) * pat.df$sd / sqrt(pat.df$n))
pat.df$lower <- pat.df$avg - (qt(0.975, df=pat.df$n) * pat.df$sd / sqrt(pat.df$n))
# ggplot(pat.df, aes(x=Patients, y=avg, colour=Method, shape=Method)) + theme_bw() + scale_colour_brewer(palette="Dark2") + 
# 		geom_point() + #geom_errorbar(aes(ymin=lower, ymax=upper), width=700) + 
# 		xlab("Number of Patients") + ylab ("Time (secs)") + scale_linetype_manual(values=c("solid", "longdash", "dotted")) + 
# 		stat_smooth(se=FALSE, aes(linetype=Method)) + theme(legend.position=c(0.15, 0.8))
# ggsave(file="patient-cpu.pdf", width=4.5, height=3.38)

########### AUC #################
rm(list=ls())
df <- read.csv("pred-model.csv", header=FALSE)
colnames(df) <- c("Sample", "Rank", "Raw", "PCA", "NMF", "Limestone", "Class")
df$id <- 1:nrow(df);
melt.df <- melt(df, id.vars=c("id", "Rank", "Sample", "Class"), variable.name="Features")
auc.df <- ddply(melt.df, .(Rank, Sample, Features), function(x) {
			pred <- prediction(x$value, x$Class)
			auc <- slot(performance(pred, "auc"), "y.values")[[1]]
			return(data.frame(Value=auc))
		})

auc.df <- ddply(auc.df, .(Rank, Features), summarise, avg=mean(Value), sd=sd(Value), n=length(Value))
auc.df$upper <- auc.df$avg + (qt(0.975, df=auc.df$n) * auc.df$sd / sqrt(auc.df$n))
auc.df$lower <- auc.df$avg - (qt(0.975, df=auc.df$n) * auc.df$sd / sqrt(auc.df$n))
raw.df <- subset(auc.df, Rank == 10 & Features == "Raw")

ggplot(subset(auc.df, Features != "Raw"), aes(x=Rank, y=avg, colour=Features, shape=Features)) +#, linetype=Features)) + 
		theme_bw() + geom_point(position=position_dodge(width=4)) + scale_colour_brewer(palette="Dark2") + #scale_linetype_manual(values=c("longdash", "longdash", "solid")) + 
		geom_line() + geom_errorbar(aes(ymin=lower, ymax=upper), position=position_dodge(width=4)) + 
		geom_hline(yintercept=raw.df$avg, colour="#535353") + scale_shape_manual(values=c(17, 15, 16)) + 
		annotate("text", 145, raw.df$avg, label="Baseline", vjust=-0.1, size=4) + 
		geom_hline(yintercept=raw.df$lower, colour="#535353", linetype="dashed") +  
		geom_hline(yintercept=raw.df$upper, colour="#535353", linetype="dashed") +  
		scale_y_continuous("AUC") + xlab("Number of Phenotypes") + theme(legend.position=c(0.87, 0.20))
ggsave(file="AUC.pdf", width=6.5, height=3.5)

### Sparsity of plots ######
rm(list=ls())
df <- read.csv("sparsity.csv", header=FALSE)
colnames(df) <- c("ID", "Mode", "Feature", "Factor", "Value")
df$Mode <- factor(df$Mode, labels=c("Diagnosis", "Medication"))
dist.df <- subset(df, ID %in% c(1) & Value > 1e-2)
d.h = hist(subset(dist.df, Mode == "Diagnosis")$Value, breaks=seq(0,1,0.01))
m.h = hist(subset(dist.df, Mode == "Medication")$Value, breaks=seq(0,1,0.01))
dist.df <- data.frame(Mode="Diagnosis", Value=d.h$breaks[-1], Count=d.h$counts)
dist.df <- rbind(dist.df, data.frame(Mode="Medication", Value=m.h$breaks[-1], Count=m.h$counts))
dist.df <- cbind(Model="Limestone", dist.df)
ggplot(subset(dist.df, Value != 0.01), aes(x=Value, y=Count, colour=Mode, shape=Mode)) + theme_bw() + #geom_histogram(binwidth=0.05, position="dodge") + 
		scale_shape_manual(values=c(17,15)) + scale_linetype_manual(values=c("solid", "longdash")) + 
		geom_line(aes(linetype=Mode)) + geom_point() + scale_y_log10("Count") + 
    scale_colour_manual(values=c("#377EB8","#E41A1C")) + xlab("Element Value") + #ylab("Count") + 
  theme(legend.position=c(0.85, 0.85))
ggsave(file="limestone-dist.pdf", width=4.5, height=3.38)

sparse.df <- subset(df, ID %in% c(101) & Value > 0.05)
sparse.df <- ddply(sparse.df, .(ID, Mode, Factor), function(x) { data.frame(Count = nrow(x)) })
sparse.df <- ddply(sparse.df, .(ID, Mode, Count), nrow)
# sparse.sum <- dcast(sparse.df, Count ~ Mode, value.var="V1")
# print(xtable(sparse.sum), include.rownames=FALSE)

ggplot(sparse.df, aes(x=Count, y=V1, colour=Mode, shape=Mode)) + theme_bw() + #geom_bar(stat="identity", position="dodge") + 
		scale_shape_manual(values=c(17,15)) + scale_linetype_manual(values=c("solid", "longdash")) + 
		geom_line(aes(linetype=Mode)) + geom_point() + 
  scale_colour_manual(values=c("#377EB8","#E41A1C")) + xlab("Nonzero Elements Per Mode") + ylab("Count") + 
  theme(legend.position=c(0.85, 0.85))
ggsave(file="sparsity.pdf", width=4.5, height=3.38)

rm(list=ls())
nmf.df <- read.csv("nmf-sparsity.csv", header=FALSE)
colnames(nmf.df) <- c("ID", "Factor", "Value")
nmf.df <- subset(nmf.df, ID == 2 & Value > 1e-4)
#dist.df <- subset(df, ID %in% c(1) & Value > 1e-2)
nmf.dist = hist(nmf.df$Value, breaks=seq(0, 1, 0.01))
nmf.dist.df <- data.frame(Value=nmf.dist$breaks[-1], Count=nmf.dist$counts)

ggplot(nmf.dist.df , aes(x=Value, y=Count)) + theme_bw() +  
  geom_line() + geom_point() + 
  scale_x_log10("Element Value") + scale_y_log10("Count") + theme(legend.position=c(0.4, 0.80))
ggsave(file="nmf-dist.pdf",  width=4.5, height=3.38)

rm(list=ls())
ptf.df <- read.csv("sparsity.csv", header=FALSE)
colnames(ptf.df) <- c("ID", "Mode", "Feature", "Factor", "Value")
ptf.df$Mode <- factor(ptf.df$Mode, labels=c("Diagnosis", "Medication"))
ptf.df <- subset(ptf.df, ID %in% c(1) & Value > 1e-2)
ptf.df <- ddply(ptf.df, .(Factor), summarise, count=length(Value))

nmf.df <- read.csv("nmf-sparsity.csv", header=FALSE)
colnames(nmf.df) <- c("ID", "Factor", "Value")
nmf.df <- subset(nmf.df, ID == 2 & Value > 1e-4)
nmf.df <- ddply(nmf.df, .(Factor), summarise, count=length(Value))

pca.df <- read.csv("fixed-pca.csv", header=FALSE, sep="|")
colnames(pca.df) <- c("ID", "Type", "Diagnosis", "Medication", "Factor", "Value")
pca.df <- ddply(pca.df, .(Factor), summarise, count=length(Value))

df <- rbind( cbind(Type="PCA", pca.df), cbind(Type="NMF", nmf.df), cbind(Type="Limestone", ptf.df))
sparse.df <- ddply(df, .(Type), summarise, avg=mean(count), median=median(count), var=var(count))
print(xtable(sparse.df), include.rownames=FALSE)

# ggplot(df, aes(x=Type, y=count, colour=Type)) + geom_boxplot(outlier.size=0) +
# 		theme_bw() + scale_y_log10("Number of Elements Per Phenotype") +
# 		xlab("Features") + scale_colour_brewer(palette="Dark2") + theme(legend.position="none")
# ggsave(file="features-count.pdf", width=4.5, height=3.38)

# sum.df <- ddply(df, .(Type), summarise, Count=mean(count))
# ggplot(sum.df, aes(x=Type, y=Count)) + geom_bar() + scale_y_log10("Count")

#threshold <- c(0.01, 0.025, 0.05);
# sparse.df <- ldply(threshold, 
# 		function(x) {
# 			tmp <- subset(df, Value > x);
# 			return(cbind(Threshold = x, ddply(tmp, .(ID, Mode, Factor), 
# 									function(y) { return (data.frame(Count=nrow(y))) } )))
# 		});
# sparse.df <- ddply(sparse.df, .(ID, Threshold, Mode, Count), nrow);
# sparse.df <- ddply(sparse.df, .(Threshold, Mode, Count), summarise, avg=mean(V1), sd=sd(V1), n=length(V1))
# ggplot(subset(sparse.df, Count <= 20), aes(x=factor(Count), y=avg, fill=Mode)) + 
# 		theme_bw() + geom_bar(stat="identity", position="dodge", alpha=1) + 
# 		scale_fill_manual(values=c("#377EB8","#E41A1C")) + facet_grid(Threshold ~ Mode) + 
# 		ylab("Count") + xlab("Nonzero Factors Per Mode") + theme(legend.position="none")
#ggsave(file="sparsity.pdf", width=10, height=5)

## Plot similarity between case and control phenotypes ###
#rm(list=ls())
#df <- read.csv("case-control.csv", header=FALSE)
#colnames(df) <- c("Mode", "T1", "T2", "Score")
#df <- subset(df, Mode >= 0)
#df$Mode <-factor(df$Mode, labels=c("Diagnosis", "Medication"))
#ggplot(df, aes(x=Score, colour=Mode)) + geom_density(aes(y=..density..), fill=NA) + 
#		theme_bw() + #geom_errorbar(score.df, aes(ymin=lower, ymax=upper)) +
#		scale_colour_manual(values=c("#377EB8","#E41A1C")) + ylab("Density") + 
#		scale_x_continuous("Score", limits=c(0,1)) + theme(legend.position=c(0.15,0.85));
#ggsave('case-control-score.pdf', width=6, height=4)

## Plot Precision/Recall Curve for meaningfulness###
rm(list=ls())
control.df <- read.csv("control-results.csv", header=TRUE)
control.df$Type = "Control"
control.df$Score = control.df$Meaningfulness
case.df <- read.csv("case-results.csv", header=TRUE)
case.df$Type = "Case"
case.df$Score <- factor(case.df$Score, labels=c("Possible", "No", "Yes"))
df <- rbind(subset(control.df, select=c("Factor", "Score", "Type")), subset(case.df, select=c("Factor", "Score", "Type")))
meaning.df <- ddply(df, .(Score, Type), nrow)
colnames(meaning.df) <- c("Annotation", "Data", "Count")
print(xtable(dcast(formula=Data~Annotation, meaning.df)), include.rownames=FALSE)

# ggplot(df, aes(x=Score, fill=Type)) + theme_bw() + #geom_histogram() + 
#  scale_fill_brewer(palette="Dark2") + geom_histogram(position="dodge") + theme(legend.position=c(0.15, 0.85)) +
#  #scale_fill_manual(values=c("#D7191C", "#A6D96A", "#1A9641")) + facet_grid(Type ~.) + theme(legend.position="none")
#  xlab("Expert Annotation") + ylab("Count") 
# ggsave(file="case-control-annotate.pdf", width=4.5, height=3.38)
#
#df <- rbind(subset(control.df, select=c("Factor", "Score", "Type")), subset(case.df, select=c("Factor", "Score", "Type")))
#df <- ddply(df, .(Type), 
#		function(x) {
#			x$tp <- cumsum(x$Score %in% c("Yes"))
#			return(x)
#		})
#df$prec <- df$tp / df$Factor
#ggplot(df, aes(x=Factor, y=prec, color=Type)) + geom_line() + theme_bw() + 
#		scale_colour_brewer(palette="Dark2") + xlab("Phenotype Number") + 
#			scale_y_continuous("Precision", limits=c(0,1)) + theme(legend.position=c(0.85,0.15))
#ggsave('case-control-prec.pdf', width=4.5, height=3.38)

## Plot similarity between NMF and Tensor
#rm(list=ls())
#df <- read.csv("sim-results.csv", header=FALSE)
#colnames(df) <- c("ID", "PTF", "NMF", "Score")
#ggplot(df, aes(x=Score, colour=factor(ID))) + geom_density(aes(y=..density..), fill=NA) + 
#		theme_bw() + scale_colour_grey() + ylab("Density") + 
#		scale_x_continuous("Score", limits=c(0,1)) + theme(legend.position="none");
#ggsave('nmf-ptf-score.pdf', width=6, height=4)

## Plot the similarity scores of the first 10 candidate phenotypes from PTF
#top10.df <- subset(df, PTF < 10)
#top10.df$PTF <- top10.df$PTF + 1
#top10.df$PTF <- factor(top10.df$PTF)
#sum10.df <- ddply(top10.df, .(PTF), summarise, avg=mean(Score), sd=sd(Score), n=length(Score))
#sum10.df$upper <- sum10.df$avg + (qt(0.975, df=sum10.df$n) * sum10.df$sd / sqrt(sum10.df$n))
#sum10.df$upper[sum10.df$upper > 1] <- 1
#sum10.df$lower <- sum10.df$avg - (qt(0.975, df=sum10.df$n) * sum10.df$sd / sqrt(sum10.df$n))
#
#ggplot(sum10.df, aes(x=PTF, y=avg)) + theme_bw() + geom_point() + 
#		geom_errorbar(aes(ymin=lower, ymax=upper, width=0.3)) + scale_y_continuous("Score", limits=c(0,1)) + xlab("Phenotype Number")
#ggsave('nmf-top10-score.pdf', width=6, height=4)
#
#rm(list=ls())
#df <- read.csv("sim-count.csv", header=FALSE, sep="|")
#df <- unique(df)
#colnames(df) <- c("Diagnosis", "Phenotype", "ID")
#temp <- ddply(df, .(Diagnosis, ID), nrow)
#sum.temp <- ddply(temp, .(Diagnosis), summarise, avg=mean(V1), n=length(V1))
#sum.temp <- sum.temp[rev(order(sum.temp$n, sum.temp$avg)),]
#write.table(sum.temp, file="disease-order.csv", sep="|")

library(plyr)
library(rjson)
cpapr.df <- read.csv('cpapr-sim.csv')
ddply(subset(cpapr.df, medication == "Beta Blockers Cardio-Selective"), .(medication), function(x) {
	total <- sum(x$value)
	return(data.frame(diagnosis = x$description, value=x$value/total))
})

med.df <- ddply(subset(cpapr.df, diagnosis == "HCC091"), .(diagnosis), function(x) {
	total <- sum(x$value)
	return(data.frame(medication=x$medication, value=x$value/total))
})
med.df <- med.df[order(med.df$value, decreasing=TRUE), ]

nmf.df <- read.csv("nmf-sim.csv")
colnames(nmf.df) <- c("HCC", "description", "medication", "value")
nmf.df$value <- nmf.df$value / sum(nmf.df$value)
subset(nmf.df, description == "")

#### GET THE COMPUTATIONAL TIME FOR THE PREDICTION TASK ########
cpu <- fromJSON(file="predPower.json")
cpuResults <- ldply(cpu, function(x) {
	return(data.frame(x))
})
## divide by 60 seconds and 60 minutes
cpuResults$comp <- cpuResults$comp / (60*60)
cpuTime <- ddply(cpuResults, .(model), summarise, Time=mean(comp))

#### GET THE NUMBER OF NON-ZEROS ########
nnzComp <- fromJSON(file="predNNZ.json")
nnzResults <- ldply(nnzComp, function(x) {
	nnz <- data.frame(nnz=x$nnz)
	return(cbind(id=x$expt, model=x$model, nnz))
})
nnzType <- ddply(nnzResults, .(model), summarise, Mean=mean(nnz), Median=median(nnz))


expt <- 10001:10009
ldpply(expt, function(x) {
	results <- read.csv(file=paste("../results/pred-db-", x, ".csv", sep=""), sep="|", header=FALSE)
	colnames(results) <- c("id", "model", "diagnosis", "medication", "factor", "value")
	bad.idx <- which(results$model == "CP-APR" & results$value < (0.05^2))
	if (length(bad.idx) > 0) { results <- results[-bad.idx, ]}
	bad.idx <- which(results$model == "NMF" & results$value < (0.05^2))
	if (length(bad.idx) > 0) { results <- results[-bad.idx, ]}
	sum.results <- ddply(subset(results, select=-c(id, diagnosis, medication)), .(model, factor), nrow)
})
