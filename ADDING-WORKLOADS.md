# Adding a Workload to gem5

This tutorial will walk you through the process of creating a Workload in gem5 and testing it, through the new gem5 Resources infrastructure introduced in gem5 v23.0.

A workload is set to a board in gem5 through the following line:

``` python
board.set_workload(Workload(<ID_OF_WORKLOAD>))
```

The Workload with ID '<ID_OF_WORKLOAD>' will be parsed and it will be used to construct the function call it defines. The function call specified in the `"function"` field of the Workload JSON is then executed on the board, along with any parameters it has defined in the `"additional_parameters"` field.


## Introduction

The gem5 Resources infrastructure allows adding a local JSON data source that can be added to the main gem5 Resources MongoDB database. We will use the local JSON data source to add a new Workload to gem5.

## Prerequisites

This tutorial assumes that you already have a pre-compiled Resource that you want to make into a Workload.

## Defining the Workload

### Defining the Resource JSON

The first step is to define the Resource that is used in a Workload. In case the Resource already exists in gem5, you may skip this step.

Let's assume that the Resource we want to wrap in a Workload is compiled for `RISC-V`, categorized as a `binary`, and has the name `binary-resource`. We can define this Resource in a JSON object as follows:

``` json
{
    "category": "binary",
    "id": "binary-resource",
    "description": "A RISCV binary used to test a specific RISCV instruction.",
    "architecture": "RISCV",
    "is_zipped": false,
    "resource_version": "1.0.0",
    "gem5_versions": [
        "23.0"
    ],
}
```

It is important to initialize all the fields here correctly, as they are used by gem5 to initialize and run the Resource.

### Defining the Workload JSON

Assuming you have the Resource JSON and the Resource is part of gem5 Resources, you can now define the Workload JSON. Let's assume that the Workload we are building wraps `binary-resource`, and is called `binary-workload`. We can define this Workload in a local JSON file as follows:

``` json
{
    "id": "binary-workload",
    "category": "workload",
    "description": "A RISCV binary used to test a specific RISCV instruction.",
    "architecture": "RISCV",
    "function": "set_se_binary_workload",
    "resource_version": "1.0.0",
    "gem5_versions": [
        "23.0"
    ],
    "resources": {
        "binary": "binary-resource"
    },
    "additional_parameters": {
        "arguments": ["arg1", "arg2"]
    }
}
```

The `"function"` field defines the function that will be called on the board. The `"resources"` field defines the Resources that will be passed into the Workload. The `"additional_parameters"` field defines the additional parameters that will be passed into the Workload. So, the Workload defined above is equivalent to the following line of code:

``` python
board.set_se_binary_workload(binary = obtain_resource("binary_resource"), arguments = ["arg1", "arg2"])
```

To see more about the fields required and not required by the workloads, see the [gem5 Resources JSON Schema](https://github.com/gem5/gem5-resources-website/blob/main/public/gem5-resources-schema.json)

## Testing the Workload

To test the Workload, we first have to add the local JSON file as a data source for gem5. This can be done by creating a new JSON file with the following format:

``` json
{
    "sources": {
        "my-resources": {
            "url": "<PATH_TO_JSON_FILE>",
            "isMongo": false,
        }
    }
}
```
On running gem5, if the JSON file is present in the current working directory, it will be used as the data source for gem5. If the JSON file is not present in the current working directory, you can specify the path to the JSON file using the `GEM5_CONFIG` flag while building gem5.

You should now be able to use the Workload in your simulations through its name, `binary-workload`.

NOTE: In order to check if the Resources you specified as part of a Workload are being passed into the Workload correctly, you can use the get_parameters() function in the Workload class. This function returns a dictionary of the Resources passed into the Workload. Its implementation can be found in [`src/python/gem5/resources/workload.py`](https://github.com/gem5/gem5/blob/af72b9ba580546ac12ce05bfaac3fd53fa8699f4/src/python/gem5/resources/workload.py#L92b).
