#!groovy
// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Docker env used for testing
// Different image may have different version tag
// because some of them are more stable than anoter.
//
// Docker images are maintained by PMC, cached in dockerhub
// and remains relatively stable over the time.
// Flow for upgrading docker env(need commiter)
//
// - Send PR to upgrade build script in the repo
// - Build the new docker image
// - Tag the docker image with a new version and push to tvmai
// - Update the version in the Jenkinsfile, send a PR
// - Fix any issues wrt to the new image version in the PR
// - Merge the PR and now we are in new version
// - Tag the new version as the lates
// - Periodically cleanup the old versions on local workers
//
ci_jekyll = "tvmai/ci-jekyll:v0.50"
ci_gpu = "tvmai/ci-gpu:v0.51"

// tvm libraries
tvm_runtime = "build/libtvm_runtime.so, build/config.cmake"
tvm_lib = "build/libtvm.so, " + tvm_runtime
// LLVM upstream lib
tvm_multilib = "build/libtvm.so, " +
             "build/libvta.so, build/libtvm_topi.so, build/libnnvm_compiler.so, " + tvm_runtime

// command to start a docker container
docker_run = 'tvm/docker/bash.sh'
// timeout in minutes
max_time = 60

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}


// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

stage("Build") {
  timeout(time: max_time, unit: 'MINUTES') {
    node('CPU') {
      ws('workspace/tvm-web/build') {
        init_git()
        sh "${docker_run} ${ci_jekyll}  ./scripts/task_build_website.sh"
        pack_lib('website', 'website.tgz')
        sh "rm -rf website.tgz"
      }
    }
  }
}

stage('Deploy') {
    node('CPU') {
      ws('workspace/tvm-web/deploy') {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('website', 'website.tgz')
           dir('_site') {
             checkout scm
             sshagent(['tvm-web']) {
               sh "git checkout asf-site"
               sh "git fetch && git reset --hard origin/asf-site"
             }
             sh "rm -rf *"
           }
           sh "tar xf website.tgz"
           dir('_site') {
             sh "git add --all && git commit -am 'nightly build'"
           }
           sshagent(['tvm-web']) {
             dir('_site') {
               sh "git push origin asf-site"
             }
           }
        }
      }
    }
}