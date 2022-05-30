# Task 3

Review the recommendations of the EU HLEG for Trustworthy AI with a specific use
case in mind. For concreteness let us consider a Computer Vision model that operates on
chest x-ray images and outputs a score indicating the presence or absence of a particular
disease (e.g. pneumothorax). Pick out three subtopics from the EU HLEG criteria (e.g.
Privacy and Data Protections) that are most important for this particular use-case from
your perspective and argue why.

## Answer

### Technical robustness and safety

- The proposed ML-System makes decisions about the presence of disease in humans, this has far reaching implications. This decision can for example be used as an argument for an operation on a patient, which carries risks for the patient. For this reason there is an ISO (ISO 16142-1:2016(en)) [3] in place to regulate devices and software used in medicine.
Therefore every point of this recommendation should be necessary.

### Transparency

- (Traceability)  
To increase the trust in the proposed ML-System, it should be possible for a third party to reproduce the decision of the system. To do this, the data set that was used to train the system has to be accessible to this third party.
- (Explainability)  
  - If the decision process of a ML-System is reproducible by a human, the human is able to learn from the system and improve the process in return.
  - If the proposed ML-System is a blackbox, the use of the system is like asking an oracle, what disease is present in the patient. (eg. bad for liability reasons(see accountability))
- (Communication)  
Patients are told how the doctor is going to treat a disease and should also be told how a disease was found. (§ 8 Aufklärungspflicht ­[2])

### Accountability

- (Redress)  
When technical systems(eg. self driving cars, anti virus programs, spam detection, ...) are in use, without human oversight the question of who is at fault for an error is often hard to answer[1].
This question becomes even harder to answer, if a ML-system by design detects problems inside a human body.  
Therefore the question of Liability and accountability has to be answered. (eg. who is at fault for an error, the ml-system, the engineer who designed the system, the entity who distributes the system or the doctor who trusted the ml-system, etc... )
- (Auditability)  
The proposed ML-System is trying to detect diseases in humans, to increase the trustworthiness of the system, the results should be reproducible by a third party.
- (Minimization and reporting of negative impacts)  
If the proposed ML-System has made an wrong judgment (eg. mislabeled a disease) there has to be a way to avoid this mislabeling in the future.

## Sources

[1] : <https://byrddavis.com/who-is-liable-when-a-self-driving-car-causes-a-crash/>  
[2] : <https://www.bundesaerztekammer.de/patienten/patientenrechte/muster-berufsordnung/>  
[3] : <https://www.iso.org/obp/ui#iso:std:iso:16142:-1:ed-1:v1:en>  
