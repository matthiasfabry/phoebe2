#ifndef PHOEBE_ERROR_HANDLING_H
	#define PHOEBE_ERROR_HANDLING_H 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "phoebe_types.h"

typedef enum PHOEBE_error_code
	{
	SUCCESS = 0,

	ERROR_SIGINT,
	ERROR_EXCEPTION_HANDLER_INVOKED,

	ERROR_HOME_ENV_NOT_DEFINED,
	ERROR_HOME_HAS_NO_PERMISSIONS,

	ERROR_FILE_NOT_FOUND,
	ERROR_FILE_NOT_REGULAR,
	ERROR_FILE_NO_PERMISSIONS,
	ERROR_FILE_IS_INVALID,
	ERROR_FILE_IS_EMPTY,
	ERROR_FILE_HAS_NO_DATA,

	ERROR_DIRECTORY_PERMISSION_DENIED,
	ERROR_DIRECTORY_TOO_MANY_FILE_DESCRIPTORS,
	ERROR_DIRECTORY_TOO_MANY_OPEN_FILES,
	ERROR_DIRECTORY_NOT_FOUND,
	ERROR_DIRECTORY_INSUFFICIENT_MEMORY,
	ERROR_DIRECTORY_NOT_A_DIRECTORY,
	ERROR_DIRECTORY_BAD_FILE_DESCRIPTOR,
	ERROR_DIRECTORY_UNKNOWN_ERROR,

	ERROR_PARAMETER_NOT_INITIALIZED,
	ERROR_PARAMETER_ALREADY_DECLARED,

	ERROR_DATA_NOT_INITIALIZED,
	ERROR_DATA_INVALID_SIZE,

	ERROR_VECTOR_NOT_INITIALIZED,
	ERROR_VECTOR_ALREADY_ALLOCATED,
	ERROR_VECTOR_IS_EMPTY,
	ERROR_VECTOR_INVALID_DIMENSION,
	ERROR_VECTOR_DIMENSIONS_MISMATCH,
	ERROR_VECTOR_DIMENSION_NOT_THREE,

	ERROR_HIST_NOT_INITIALIZED,
	ERROR_HIST_NOT_ALLOCATED,
	ERROR_HIST_ALREADY_ALLOCATED,
	ERROR_HIST_INVALID_DIMENSION,
	ERROR_HIST_INVALID_RANGES,
	ERROR_HIST_NO_OVERLAP,
	ERROR_HIST_OUT_OF_RANGE,

	ERROR_ARRAY_NOT_INITIALIZED,
	ERROR_ARRAY_ALREADY_ALLOCATED,
	ERROR_ARRAY_INVALID_DIMENSION,

	ERROR_CURVE_NOT_INITIALIZED,
	ERROR_CURVE_ALREADY_ALLOCATED,
	ERROR_CURVE_INVALID_DIMENSION,

	ERROR_SPECTRUM_NOT_INITIALIZED,
	ERROR_SPECTRUM_NOT_ALLOCATED,
	ERROR_SPECTRUM_ALREADY_ALLOCATED,
	ERROR_SPECTRUM_INVALID_DIMENSION,
	ERROR_SPECTRUM_INVALID_SAMPLING,
	ERROR_SPECTRUM_INVALID_RESOLUTION,
	ERROR_SPECTRUM_INVALID_RANGE,
	ERROR_SPECTRUM_NOT_IN_REPOSITORY,

	ERROR_INDEX_OUT_OF_RANGE,

	ERROR_INVALID_TYPE,
	ERROR_INVALID_INDEP,
	ERROR_INVALID_DEP,
	ERROR_INVALID_WEIGHT,
	ERROR_INVALID_DATA,
	ERROR_INVALID_MODEL,
	ERROR_INVALID_LDLAW,
	ERROR_INVALID_EL3_UNITS,
	ERROR_INVALID_EL3_VALUE,
	ERROR_INVALID_MAIN_INDEP,
	ERROR_INVALID_HEADER,
	ERROR_INVALID_WAVELENGTH_INTERVAL,
	ERROR_INVALID_SAMPLING_POWER,

	ERROR_UNINITIALIZED_CURVE,

	ERROR_UNSUPPORTED_MPAGE,

	ERROR_MINIMIZER_FEEDBACK_NOT_INITIALIZED,
	ERROR_MINIMIZER_FEEDBACK_ALREADY_ALLOCATED,
	ERROR_MINIMIZER_NO_CURVES,
	ERROR_MINIMIZER_NO_PARAMS,
	ERROR_MINIMIZER_INVALID_FILE,
	ERROR_MINIMIZER_DPDT_WITH_PHASES,
	ERROR_MINIMIZER_HLA_REQUEST_NOT_SANE,
	ERROR_MINIMIZER_VGA_REQUEST_NOT_SANE,
	ERROR_MINIMIZER_DPDT_REQUEST_NOT_SANE,

	ERROR_NONSENSE_DATA_REQUEST,
	ERROR_NEGATIVE_STANDARD_DEVIATION,

	ERROR_CHI2_INVALID_TYPE,
	ERROR_CHI2_INVALID_DATA,
	ERROR_CHI2_DIFFERENT_SIZES,

	ERROR_MS_TEFF1_OUT_OF_RANGE,
	ERROR_MS_TEFF2_OUT_OF_RANGE,

	ERROR_PLOT_DIMENSION_MISMATCH,
	ERROR_PLOT_FIFO_PERMISSION_DENIED,
	ERROR_PLOT_FIFO_FILE_EXISTS,
	ERROR_PLOT_FIFO_FAILURE,
	ERROR_PLOT_TEMP_FILE_EXISTS,
	ERROR_PLOT_TEMP_FAILURE,
	ERROR_PLOT_TEMP_MALFORMED_FILENAME,

	ERROR_GSL_NOT_INSTALLED,
	ERROR_GNUPLOT_NOT_INSTALLED,

	ERROR_LD_PARAMS_OUT_OF_RANGE,
	ERROR_LD_TABLES_MISSING,
	ERROR_LD_LAW_INVALID,

	ERROR_CINDEX_INVALID_TYPE,

	ERROR_PARAMETER_INDEX_OUT_OF_RANGE,
	ERROR_PARAMETER_KIND_NOT_MENU,
	ERROR_PARAMETER_MENU_ITEM_OUT_OF_RANGE,
	ERROR_PARAMETER_INVALID_LIMITS,

	ERROR_PASSBAND_TF_FILE_NOT_FOUND,
	ERROR_PASSBAND_INVALID,

	ERROR_QUALIFIER_NOT_FOUND,
	ERROR_QUALIFIER_NOT_ADJUSTABLE,
	ERROR_QUALIFIER_NOT_ARRAY,

	ERROR_DESCRIPTION_NOT_FOUND,

	ERROR_ARG_NOT_INT,
	ERROR_ARG_NOT_BOOL,
	ERROR_ARG_NOT_DOUBLE,
	ERROR_ARG_NOT_STRING,
	ERROR_ARG_NOT_INT_ARRAY,
	ERROR_ARG_NOT_BOOL_ARRAY,
	ERROR_ARG_NOT_DOUBLE_ARRAY,
	ERROR_ARG_NOT_STRING_ARRAY,

	ERROR_ARG_NUMBER_MISMATCH,

	ERROR_RELEASE_INDEX_OUT_OF_RANGE,

	ERROR_SPECTRA_REPOSITORY_NOT_FOUND,
	ERROR_SPECTRA_REPOSITORY_INVALID_NAME,
	ERROR_SPECTRA_REPOSITORY_EMPTY,
	ERROR_SPECTRA_REPOSITORY_NOT_DIRECTORY,

	ERROR_SPECTRA_DIMENSION_MISMATCH,
	ERROR_SPECTRA_NO_OVERLAP,
	ERROR_SPECTRA_NOT_ALIGNED,

	ERROR_BROADENING_INADEQUATE_ACCURACY
	} PHOEBE_error_code;

char *phoebe_error       (PHOEBE_error_code code);
int   phoebe_lib_error   (const char *fmt, ...);
int   phoebe_lib_warning (const char *fmt, ...);
int   phoebe_debug       (const char *fmt, ...);

void *phoebe_malloc (size_t size);
void *phoebe_realloc (void *ptr, size_t size);

bool hla_request_is_sane ();
bool vga_request_is_sane ();
bool dpdt_request_is_sane ();

#endif
