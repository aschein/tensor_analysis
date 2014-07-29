require(ggplot2)
require(reshape2)
require(plyr)

extract.rank.sample <- function(data) {
	data$Sample <- "HF";
	data$Sample[data$Model == 3] <- 'Subsampled Control';
	data$Sample <- factor(data$Sample);
	return(data);
}

################# Generate the plots for rank as a function of CPU Time #################
rank.time <- read.csv('rank-time.csv', sep='|', header=FALSE)
colnames(rank.time) <- c("Model", "Rank", "Iteration", "Mode", "time")
rank.time$Mode <- factor(rank.time$Mode)
rank.time <- extract.rank.sample(rank.time)

ggplot(rank.time, aes(x=Rank, y=time, colour=Sample)) + theme_bw() + scale_color_brewer(palette="Set1") + 
		geom_point()+ stat_smooth() + facet_grid(Mode ~., scales="free") + theme(legend.position='top') + 
		xlab("Factorization Rank (k)") + ylab("Time (secs)") + ggtitle("Rank versus CPU Time")
ggsave(file="rank-cpu.pdf", width=8, height=12)
rm(rank.time)

################# Generate the plots for rank as a function of log likelihood #################
#rank.df <- read.csv('rank-results.csv', sep='|', header=FALSE)
#colnames(rank.df) <- c("Model", "Label ID", "Patient Set", "Outer", "Inner", "Rank", 
#		"Least Squares", "Log Likelihood", "KKT")
#rank.df <- extract.rank.sample(rank.df)
#
#melt.df <- melt(subset(rank.df, select=c("Sample", "Rank", "Least Squares", "Log Likelihood")), id.vars=c("Sample", "Rank"))
#ggplot(melt.df, aes(x=Rank, y=value, colour=Sample)) + theme_bw() + scale_color_brewer(palette="Set1") + 
#		geom_point() + geom_line() + facet_grid(variable ~ ., scales="free") + theme(legend.position='top') +
#		ggtitle("Rank versus Accuracy")
#ggsave(file="rank-acc.pdf", width=8, height=8)
#rm(rank.df, melt.df)

################# Generate the plots to see how fast convergence occurs #################
iter.df <- read.csv('iter-results.csv', sep='|', header=FALSE)
colnames(iter.df) <- c('Model', "Iteration", "Mode", "Inner Iterations", "Least Squares", "Log Likelihood", "KKT", "Time")
iter.df$Model <- factor(iter.df$Model)
iter.df$Mode <- factor(iter.df$Mode)
iter.df <- iter.df[order(iter.df$Model, iter.df$Iteration, iter.df$Mode), ]
#ignore the modes
iter.nomodes <- ddply(iter.df, .(Model, Iteration), tail, n=1)
single.iter <- melt(subset(iter.nomodes, Model == 5, select=c("Model", "Iteration", "Least Squares", "Log Likelihood")), id.vars=c("Model", "Iteration"))
iter.fms.df <- read.csv('iter-fms.csv', sep='|', header=FALSE);
colnames(iter.fms.df) <- c("Model", "Iteration", "Top 10", "Greedy");
single.fms.iter <- melt(subset(iter.fms.df, Model == 5), id.vars=c("Model", "Iteration"))
single.df <- rbind(single.iter, single.fms.iter)
ggplot(single.df, aes(x=Iteration, y=value)) + theme_bw() + scale_color_brewer(palette="Set1") + 
		geom_point() + facet_grid(variable ~ ., scales="free")
rm(list=ls())



################# Generate the plots for uniqueness as a function of iterations #################
extract.hierarchy <- function(data) {
	data <- subset(data, Model >= 600 | Model < 500)
	data$Axis <- "Pharmacy / HCC Hierarchy"
	data$Axis[data$Model < 200] <- "Pharmacy Subclass / HCC"
	data$Axis[data$Model >= 600] <- "Combo"
	data$Axis <- factor(data$Axis);
	return(data);
}

uniq.df <- read.csv('uniq-results.csv', sep='|', header=FALSE)
colnames(uniq.df) <- c("Model", "Initial", "Iterations", "Least Squares", "Log Likelihood")
uniq.fms <- read.csv('uniq-fms.csv', sep='|', header=FALSE)
colnames(uniq.fms) <- c("Model", "Iterations", "InitA", "InitB", "Top 10 Score", "Greedy Score")
uniq.fms$Initial <- 1:nrow(uniq.fms)

level1.df <- subset(uniq.df, Model < 200)
level1.melt <- melt(subset(level1.df, select=-c(Model)), id.vars=c("Initial", "Iterations"))
level1.melt$Initial <- factor(level1.melt$Initial)
ggplot(level1.melt, aes(x=Iterations, y=value, colour=Initial)) + scale_colour_brewer(palette="Paired") + 
		theme_bw() + geom_point() + geom_line() + facet_grid(variable ~., scales="free") +
		ylab("Value") + xlim(0, 150) +theme(legend.position="none")
ggsave(file="uniq-level1.pdf", width=6, height=8)


level1.fms <- subset(uniq.fms, Model < 200)
level1.fms.melt <- melt(subset(level1.fms, select=-c(Model, InitA, InitB)), id.vars=c("Initial", "Iterations"))
level1.df <- rbind(level1.melt, level1.fms.melt)
ggplot(level1.df, aes(x=Iterations, y=value)) + theme_bw() + 
		scale_color_brewer(palette="Set1") + geom_point() + 
		facet_grid(variable ~., scales="free") + xlim(0, 150) +
		theme(legend.position="top") + ggtitle("Uniqueness vs Iterations")

uniq.df <- extract.hierarchy(uniq.df);
uniq.melt <- melt(subset(uniq.df, select=-c(Model)), id.vars=c("Axis", "Initial", "Iterations"))

uniq.fms <- extract.hierarchy(uniq.fms)
fms.melt <- melt(subset(uniq.fms, select=-c(Model, InitA, InitB)), id.vars=c("Axis","Initial", "Iterations"))

melt.df <- rbind(uniq.melt, fms.melt)
ggplot(melt.df, aes(x=Iterations, y=value, colour=Axis)) + theme_bw() + 
		scale_color_brewer(palette="Set1") + geom_point(size=1.5, position=position_jitter(w=0.4, h=0)) + 
		stat_smooth() + facet_grid(variable ~., scales="free") + xlim(0, 150) +
		theme(legend.position="top") + ggtitle("Uniqueness vs Iterations")
ggsave(file="uniq-iter.pdf", width=8, height=12)
