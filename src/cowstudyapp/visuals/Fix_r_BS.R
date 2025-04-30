# Fix_r_BS.R - Simplified installation script
options(repos = c(CRAN = "https://cran.rstudio.com/"))

cat("Installing Matrix 1.5-3...\n")
install.packages("https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-3.tar.gz", 
                repos = NULL, type = "source")

cat("Loading Matrix to verify installation...\n")
library(Matrix)
cat(sprintf("Matrix version: %s\n", packageVersion("Matrix")))

cat("Installing lme4 package...\n")
install.packages("lme4", dependencies = TRUE)

cat("Loading lme4 to verify installation...\n")
library(lme4)
cat(sprintf("lme4 version: %s\n", packageVersion("lme4")))

cat("Installing additional packages...\n")
packages <- c("lmerTest", "pbkrtest", "MASS", "numDeriv")
install.packages(packages)

cat("\nVerifying all installations:\n")
for (pkg in c("Matrix", "lme4", packages)) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("%s: %s\n", pkg, packageVersion(pkg)))
  } else {
    cat(sprintf("%s: NOT INSTALLED\n", pkg))
  }
}

cat("\nInstallation complete!\n")