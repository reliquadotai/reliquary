# Local Ansible inventory

The public repository contains examples only. Create the ignored operational
files before running either playbook:

~~~bash
cp inventory/hosts.example.yml inventory/hosts.yml
cp inventory/group_vars/all.example.yml inventory/group_vars/all.yml
cp inventory/known_hosts.example inventory/known_hosts
~~~

Replace every example value locally. Add the server's OpenSSH host-key entry to
`inventory/known_hosts` only after comparing its fingerprint with the value
received through the hosting provider or another independent channel. Never
commit these three operational files.
