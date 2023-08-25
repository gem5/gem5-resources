import os
import re
import shutil
import subprocess
import shlex
import logging
import random
import string
from string import Template
import sys

import riscof.utils as utils
import riscof.constants as constants
from riscof.pluginTemplate import pluginTemplate

logger = logging.getLogger()

class gem5(pluginTemplate):
    __model__ = "gem5"

    __version__ = "23.0.1.0"

    def __init__(self, *args, **kwargs):
        sclass = super().__init__(*args, **kwargs)

        config = kwargs.get('config')

        if config is None:
            logger.error("Please enter input file paths in configuration.")
            raise SystemExit
        try:
            self.isa_spec = os.path.abspath(config['ispec'])
            self.platform_spec = os.path.abspath(config['pspec'])
            self.pluginpath = os.path.abspath(config['pluginpath'])
            self.gem5path = os.path.abspath(config['gem5path'])
            self.configfile = os.path.abspath(config['configpath'])
            self.configargs = config['configargs']
            self.make = config['make'] if 'make' in config else 'make'
        except KeyError as e:
            logger.error("Please check the gem5 section in config for missing values.")
            logger.error(e)
            raise SystemExit

        try:
            if config.get('required_build'):
                self.required_build = eval(config['required_build'])
            else:
                self.required_build = False
        except ValueError as e:
            self.required_build = False

        try:
            self.docker = eval(config['docker']) if 'docker' in config else False
        except Exception:
            self.docker = False
        finally:
            if not isinstance(self.docker, bool):
                self.docker = False

        if self.docker:
            try:
                self.docker_img = str(config['image'])
            except KeyError as e:
                logger.error("Please specify the image for build & run gem5 binary.")
                logger.error(e)
                raise SystemExit

        self.build_opts = config['build_opts'] if 'build_opts' in config else 'RISCV'
        self.dut_exe = os.path.join(self.gem5path, f"build/{self.build_opts}/gem5.opt")
        self.num_jobs = str(config['jobs'] if 'jobs' in config else 1)

        logger.debug("Gem5 plugin initialised using the following configuration.")
        for entry in config:
            logger.debug(entry+' : '+config[entry])
        return sclass

    def initialise(self, suite, work_dir, archtest_env):
        self.work_dir = work_dir

        #TODO: The following assumes you are using the riscv-gcc toolchain. If
        #      not please change appropriately
        self.compile_cmd = 'riscv{1}-unknown-elf-gcc -march={0} \
         -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles\
         -T '+self.pluginpath+'/env/link.ld\
         -I '+self.pluginpath+'/env/\
         -I ' + archtest_env

        # set all the necessary variables like compile command, elf2hex
        # commands, objdump cmds. etc whichever you feel necessary and required
        # for your plugin.

    def build(self, isa_yaml, platform_yaml):
        ispec = utils.load_yaml(isa_yaml)['hart0']
        self.xlen = ('64' if 64 in ispec['supported_xlen'] else '32')
        self.isa = 'rv' + self.xlen
        #TODO: The following assumes you are using the riscv-gcc toolchain. If
        #      not please change appropriately
        if "64I" in ispec["ISA"]:
            self.compile_cmd = self.compile_cmd+' -mabi='+'lp64 '
        elif "64E" in ispec["ISA"]:
            self.compile_cmd = self.compile_cmd+' -mabi='+'lp64e '
        elif "32I" in ispec["ISA"]:
            self.compile_cmd = self.compile_cmd+' -mabi='+'ilp32 '
        elif "32E" in ispec["ISA"]:
            self.compile_cmd = self.compile_cmd+' -mabi='+'ilp32e '
        self.isa = ispec["ISA"].lower()

        compiler = "riscv{0}-unknown-elf-gcc".format(self.xlen)
        if shutil.which(compiler) is None:
            logger.error(compiler+": executable not found. Please check environment setup.")
            raise SystemExit
        if not os.path.exists(self.dut_exe):
            self.required_build = True
            if not self.docker and shutil.which('scons') is None:
                logger.error("scons: executable not found. Please check environment setup.")
                raise SystemExit
        if shutil.which(self.make) is None:
            logger.error(self.make+": executable not found. Please check environment setup.")
            raise SystemExit
        if shutil.which('hexdump') is None:
            logger.error("hexdump: executable not found. Please check environment setup.")
            raise SystemExit

        if self.required_build:
            utils.shellCommand(f'rm -rf build/{self.build_opts}').run(cwd=self.gem5path)
            build_shell_cmd = 'scons --ignore-style -j' + self.num_jobs + \
                               ' ' + f'build/{self.build_opts}/gem5.opt'
            if self.docker:
                docker_shell_cmd = (
                    f"docker run -u {os.getuid()}:{os.getgid()} --volume "
                    f"'{self.gem5path}':'{self.gem5path}' -w '{self.gem5path}' "
                    f"--rm {self.docker_img} "
                )
                build_cmd = docker_shell_cmd + build_shell_cmd
            else:
                build_cmd = build_shell_cmd

            utils.shellCommand(build_cmd).run(cwd=self.gem5path, timeout=3600)

    def runTests(self, testList, cgf_file=None):
        make = utils.makeUtil(makefilePath=os.path.join(self.work_dir, "Makefile." + self.name[:-1]))
        make.makeCommand = self.make + ' -j' + self.num_jobs
        for file in testList:
            testentry = testList[file]
            test = testentry['test_path']
            test_dir = testentry['work_dir']

            if cgf_file is not None:
                elf = 'ref.elf'
            else:
                elf = 'dut.elf'

            execute = "@cd "+testentry['work_dir']+";"

            cmd = self.compile_cmd.format(testentry['isa'].lower(), self.xlen) + ' ' + test + ' -o ' + elf

            #TODO: we are using -D to enable compile time macros. If your
            #      toolchain is not riscv-gcc you may want to change the below code
            compile_cmd = cmd + ' -D' + " -D".join(testentry['macros'])
            execute+=compile_cmd+";touch signature;"

            sig_file = os.path.join(test_dir, self.name[:-1] + ".signature")
            sig_binary = os.path.join(test_dir, 'signature')

            run_shell_cmd = self.dut_exe + ' -d . -re --silent-redirect '  + \
                            self.configfile + ' ' + \
                            self.configargs.format(elf, self.xlen) + ';'
            if self.docker:
                docker_shell_cmd = (
                    f"docker run -u {os.getuid()}:{os.getgid()} --volume "
                    f"'{self.gem5path}':'{self.gem5path}' "
                    f"--volume '{test_dir}':'{test_dir}' -w '{test_dir}' "
                    f"--rm {self.docker_img} "
                )
                execute += docker_shell_cmd
            execute += run_shell_cmd

            hexdump_cmd = "hexdump -v -e '1/4 \"%08x\\n\"' {1} > {0};".format(
                    sig_file, sig_binary)
            execute+=hexdump_cmd

            # gem5 doesn't support coverage
            coverage_cmd = ''
            execute+=coverage_cmd

            make.add_target(execute)
        make.execute_all(self.work_dir)
