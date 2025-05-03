# CSE152B - Spring 2025: Homework 3
## Homework instructions

The homework is in the Jupyter Notebook ``HW3.ipynb``.

1. Attempt all questions.
2. Please comment all your code adequately.
3. Include all relevant information such as text answers, output images in notebook.
4. **Academic integrity:** The homework must be completed individually.

5. **Submission instructions:**  
 (a) Submit the notebook and its PDF version on Gradescope.  
 (b) To save as PDF, use Ctrl + P -> Save as PDF (toggling Headers and footers, Background graphics).  
 (c) Rename your submission files as Lastname_Firstname.ipynb and Lastname_Firstname.pdf.  
 (d) Correctly select pages for each answer on Gradescope to allow proper grading.

6. **Due date:** Assignments are due X, by 11.59pm PST.

## Extra Instructions

### Fetch output files from the cluster

From your local machine: 

``scp -r <USERNAME>@dsmlp-login.ucsd.edu:{path to files on the cluster} {LOCAL PATH}``

### Attach the container from your terminal

Once you launch a container from the JupyterHub portal, you can also access the container with command line from your local terminal:

- `ssh {your ucsd id}@dsmlp-login.ucsd.edu` # use your UCSD credentials; UCSD VPN connectin may be needed
- get active container via `kubectl get pods`
- attach to the pod via `kubesh {pod name you got from above}`
- then you will be in the bash environment inside the container, identical to the terminal launched from Jupyter Notebook.
### Launch the container from your terminal

Alternatively, you can launch a container solely from commandline. Follow those steps:

- `ssh {your ucsd id}@dsmlp-login.ucsd.edu` # use your UCSD credentials; UCSD VPN connectin may be needed
- Launch your pod.
  - You should enter a node with 1 GPU, 4 CPU, 16 GB RAM, with normal priority (running up to 6 hours
    - `launch-scipy-ml.sh -i ucsdets/cse152-252-notebook:latest -g 1 -m 16 -c 4 -p normal`
  - To enable longer runtime k (up to 12) hours with normal priority
    - `K8S_TIMEOUT_SECONDS=$((3600*k)) launch-scipy-ml.sh -i ucsdets/cse152-252-notebook:latest -g 1 -m 16 -c 4 -p normal`
  - To enable longer runtime k (more than 12) hours with lower priority
    - `K8S_TIMEOUT_SECONDS=$((3600*k)) launch-scipy-ml.sh -i ucsdets/cse152-252-notebook:latest -g 1 -m 16 -c 4`
  - To run your container in the background up to 12 hours, add -b to above command. See details [here](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=c72a818f1b8e6050df40ed7dee4bcb31).
- You will be provided with a URL that you can open locally. Click on the link and navigate to the Jupyter notebook.
- If you cannot launch a pod, set up the environment following these [instructions](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=cbb951c31b42a050df40ed7dee4bcb9e).

### Maintain a session within a container
There are cases that you may want to maintain your current session (e.g. a Python training job) within the container when you need to go offline. You can achieve this with session managers like 
`tmux`.

For quick start,
- Just run ``tmux`` on your terminal once you get into the container.
- To detach and come back later, use `ctrl + b` then `d`. To attach next time, use `ctrl + b` then `a`.
- For more TMUX usages please refer to online tutorials like [https://linuxize.com/post/getting-started-with-tmux/](https://linuxize.com/post/getting-started-with-tmux/) ot this [post](https://leimao.github.io/blog/Tmux-Tutorial/)
