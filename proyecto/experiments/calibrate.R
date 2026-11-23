args <- commandArgs(TRUE)
method <- args[1]; python <- args[2]; root <- args[3]; library_path <- args[4]
out <- args[5]; max_calls <- as.integer(args[6]); kappa <- as.integer(args[7])
.libPaths(c(library_path, .libPaths()))
suppressPackageStartupMessages(library(irace))
dir.create(out, recursive=TRUE, showWarnings=FALSE)
setwd(root)
pop <- 'population "" c (20,30,50,80)'
common <- c(pop,'alpha "" r (0.05,0.50)','discount "" r (0.50,0.95)',
            'epsilon "" r (0.10,0.50)','tau "" r (20,100)')
definitions <- switch(method,
 ga=c(pop,'crossover "" r (0.60,0.95)','mutation "" r (0.05,0.50)'),
 dpso=c(pop,'w "" r (0.3,0.9)','c1 "" r (0.5,2.5)','c2 "" r (0.5,2.5)'),
 sboa=pop, dmshoa=c(pop,'k "" r (0.1,0.8)'), uniform=pop,
 fixed=c(pop,sprintf('w%d "" r (0.05,1.0)',0:6)),
 qlma=common, no_diversification=common, stop('unknown method'))
writeLines(definitions,file.path(out,'parameters.txt'))
parameters <- readParameters(text=paste(definitions,collapse='\n'))
calls <- 0L
target <- function(experiment, scenario) {
  calls <<- calls + 1L
  configuration <- experiment$configuration
  parameter_file <- file.path(out,sprintf('configuration-%05d.csv',calls))
  write.csv(configuration,parameter_file,row.names=FALSE)
  seed <- 100000000L + (as.integer(experiment$seed) %% 100000000L)
  cmd <- c(shQuote(file.path(root,'experiments/research.py')),
           '--target',method,'--instance',shQuote(experiment$instance),
           '--seed',as.character(seed),'--parameters',shQuote(parameter_file),
           '--kappa',as.character(kappa),'--record-dir',shQuote(file.path(out,'calls')))
  output <- system2(python,cmd,stdout=TRUE,stderr=file.path(out,'target-errors.log'))
  status <- attr(output,'status')
  cost <- suppressWarnings(as.numeric(tail(output,1)))
  if ((!is.null(status) && status!=0L) || length(cost)!=1L || !is.finite(cost))
    stop(paste('target failed:',paste(output,collapse=' ')))
  cat(sprintf('call=%d method=%s cost=%.9f\n',calls,method,cost))
  flush.console()
  list(cost=cost)
}
scenario <- defaultScenario()
scenario$parameters <- parameters
scenario$targetRunner <- target
scenario$instances <- file.path(root,c('instances/standard/HOSP-STD-20-01.yaml',
                                     'instances/hospital_12rooms/HOSP-12R-40-01.yaml'))
scenario$maxExperiments <- max_calls
scenario$seed <- 71001L + match(method,c('ga','dpso','sboa','dmshoa','fixed','uniform','qlma','no_diversification'))
scenario$parallel <- 0L
scenario$logFile <- file.path(out,'irace.Rdata')
scenario$quiet <- TRUE
scenario$debugLevel <- 0L
if (max_calls<500L) {
  scenario$nbConfigurations <- 6L
  scenario$nbIterations <- 2L
  scenario$minNbSurvival <- 2L
  scenario$firstTest <- 2L
}
selected <- irace(scenario)
write.csv(selected[1,,drop=FALSE],file.path(out,'selected.csv'),row.names=FALSE)
writeLines(c(paste0('irace=',packageVersion('irace')),paste0('R=',getRversion()),
             paste0('calls=',calls),paste0('kappa=',kappa)),file.path(out,'completed.txt'))
