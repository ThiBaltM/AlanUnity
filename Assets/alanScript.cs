using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;

public class alanScript : Agent
{
    public GameObject head;
    public GameObject leftHeel;
    public GameObject leftTibia;
    public GameObject leftFeet;

    public GameObject rightHeel;
    public GameObject rightTibia;
    public GameObject rightFeet;

    private HingeJoint leftHeelJoint;
    private HingeJoint leftTibiaJoint;
    private HingeJoint leftFeetJoint;

    private HingeJoint rightHeelJoint;
    private HingeJoint rightTibiaJoint;
    private HingeJoint rightFeetJoint;


    //initial position
    // Variables pour sauvegarder les positions et rotations initiales
    private Vector3 headInitialPosition;
    private Quaternion headInitialRotation;
    private Vector3 leftHeelInitialPosition;
    private Quaternion leftHeelInitialRotation;
    private Vector3 leftTibiaInitialPosition;
    private Quaternion leftTibiaInitialRotation;
    private Vector3 leftFeetInitialPosition;
    private Quaternion leftFeetInitialRotation;
    private Vector3 rightHeelInitialPosition;
    private Quaternion rightHeelInitialRotation;
    private Vector3 rightTibiaInitialPosition;
    private Quaternion rightTibiaInitialRotation;
    private Vector3 rightFeetInitialPosition;
    private Quaternion rightFeetInitialRotation;

    // Start is called before the first frame update
    void Start()
    {
        leftHeelJoint = leftHeel.GetComponent<HingeJoint>();
        leftTibiaJoint = leftTibia.GetComponent<HingeJoint>();
        leftFeetJoint = leftFeet.GetComponent<HingeJoint>();

        rightHeelJoint = rightHeel.GetComponent<HingeJoint>();
        rightTibiaJoint = rightTibia.GetComponent<HingeJoint>();
        rightFeetJoint = rightFeet.GetComponent<HingeJoint>();

        // Sauvegarder les positions et rotations initiales
        headInitialPosition = head.transform.position;
        headInitialRotation = head.transform.rotation;
        leftHeelInitialPosition = leftHeel.transform.position;
        leftHeelInitialRotation = leftHeel.transform.rotation;
        leftTibiaInitialPosition = leftTibia.transform.position;
        leftTibiaInitialRotation = leftTibia.transform.rotation;
        leftFeetInitialPosition = leftFeet.transform.position;
        leftFeetInitialRotation = leftFeet.transform.rotation;
        rightHeelInitialPosition = rightHeel.transform.position;
        rightHeelInitialRotation = rightHeel.transform.rotation;
        rightTibiaInitialPosition = rightTibia.transform.position;
        rightTibiaInitialRotation = rightTibia.transform.rotation;
        rightFeetInitialPosition = rightFeet.transform.position;
        rightFeetInitialRotation = rightFeet.transform.rotation;
    }


    public override void OnEpisodeBegin()
    {
        // Réinitialiser les positions et rotations des segments
        head.transform.position = headInitialPosition;
        head.transform.rotation = headInitialRotation;
        leftHeel.transform.position = leftHeelInitialPosition;
        leftHeel.transform.rotation = leftHeelInitialRotation;
        leftTibia.transform.position = leftTibiaInitialPosition;
        leftTibia.transform.rotation = leftTibiaInitialRotation;
        leftFeet.transform.position = leftFeetInitialPosition;
        leftFeet.transform.rotation = leftFeetInitialRotation;
        rightHeel.transform.position = rightHeelInitialPosition;
        rightHeel.transform.rotation = rightHeelInitialRotation;
        rightTibia.transform.position = rightTibiaInitialPosition;
        rightTibia.transform.rotation = rightTibiaInitialRotation;
        rightFeet.transform.position = rightFeetInitialPosition;
        rightFeet.transform.rotation = rightFeetInitialRotation;

        // Réinitialiser les angles des articulations
        ResetJointAngles(leftHeelJoint);
        ResetJointAngles(leftTibiaJoint);
        ResetJointAngles(leftFeetJoint);
        ResetJointAngles(rightHeelJoint);
        ResetJointAngles(rightTibiaJoint);
        ResetJointAngles(rightFeetJoint);
    }

    void ResetJointAngles(HingeJoint joint)
    {
        JointSpring spring = joint.spring;
        spring.targetPosition = 0; // Position initiale
        joint.spring = spring;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(head.transform.position);
        Debug.Log($"Position de la tête: {head.transform.position}");

        // Angles des articulations pour la jambe gauche
        sensor.AddObservation(leftHeelJoint.angle);
        sensor.AddObservation(leftTibiaJoint.angle);
        sensor.AddObservation(leftFeetJoint.angle);
        Debug.Log($"Angles de la jambe gauche: Heel {leftHeelJoint.angle}, Tibia {leftTibiaJoint.angle}, Feet {leftFeetJoint.angle}");

        // Angles des articulations pour la jambe droite
        sensor.AddObservation(rightHeelJoint.angle);
        sensor.AddObservation(rightTibiaJoint.angle);
        sensor.AddObservation(rightFeetJoint.angle);
        Debug.Log($"Angles de la jambe droite: Heel {rightHeelJoint.angle}, Tibia {rightTibiaJoint.angle}, Feet {rightFeetJoint.angle}");

    }

    public override void OnActionReceived(ActionBuffers actionBuffers){
        // Appliquer les actions de l'IA aux joints
        float leftHeelTarget = actionBuffers.ContinuousActions[0];
        float leftTibiaTarget = actionBuffers.ContinuousActions[1];
        float leftFeetTarget = actionBuffers.ContinuousActions[2];

        float rightHeelTarget = actionBuffers.ContinuousActions[0];
        float rightTibiaTarget = actionBuffers.ContinuousActions[1];
        float rightFeetTarget = actionBuffers.ContinuousActions[2];

        SetJointTarget(leftHeelJoint, leftHeelTarget);
        SetJointTarget(leftTibiaJoint, leftTibiaTarget);
        SetJointTarget(leftFeetJoint, leftFeetTarget);

        SetJointTarget(rightHeelJoint, rightHeelTarget);
        SetJointTarget(rightTibiaJoint, rightTibiaTarget);
        SetJointTarget(rightFeetJoint, rightFeetTarget);
    }

    void SetJointTarget(HingeJoint joint, float target)    {
            JointSpring spring = joint.spring;
            spring.targetPosition = target;
            joint.spring = spring;
    }

    void Update()
    {
        // Demander des décisions et actions à chaque frame
        RequestDecision();
        RequestAction();
    }

}
